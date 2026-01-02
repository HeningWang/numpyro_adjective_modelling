import os
import argparse
#from IPython.display import set_matplotlib_formats
import jax
import jax.numpy as jnp
from jax import random, vmap
from jax.scipy.special import logsumexp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import math
import scipy
import arviz as az
import numpyro
from numpyro.diagnostics import hpdi
import numpyro.distributions as dist
from numpyro import handlers
from numpyro.infer import MCMC, NUTS, Predictive
from jax import lax
from sklearn.model_selection import train_test_split
from helper import link_function
from jax.random import PRNGKey
numpyro.set_platform("cpu")

print(jax.__version__)
jax.devices()

def get_midrange_extrema_from_context(
    states: jnp.ndarray,
    state_prior: jnp.ndarray,
    q_low: float = 0.2,
    q_high: float = 0.8,
) -> tuple[float, float]:
    """
    Compute mid-range extrema x_min^mid(C), x_max^mid(C) from the context C.

    Args:
        states:      (n_obj, 3) array; size is states[:, 0].
        state_prior: (n_obj,) array; contextual distribution C(s).
        q_low:       lower quantile for mid-range band (0 < q_low < q_high < 1).
        q_high:      upper quantile for mid-range band.

    Returns:
        x_min_mid, x_max_mid: floats defining the mid-range band.
    """
    sizes = states[:, 0]  # shape (n_obj,)

    # Sort by size
    idx = jnp.argsort(sizes)
    sizes_sorted = sizes[idx]
    prior_sorted = state_prior[idx]

    # CDF of the contextual prior over sorted sizes
    cdf = jnp.cumsum(prior_sorted)

    def quantile_index(q):
        # First index where CDF >= q
        # (works because cdf[-1] = 1, so mask is True somewhere)
        mask = cdf >= q
        return jnp.argmax(mask)

    i_low = quantile_index(q_low)
    i_high = quantile_index(q_high)

    x_min_mid = sizes_sorted[i_low]
    x_max_mid = sizes_sorted[i_high]

    return x_min_mid, x_max_mid

def get_threshold_k_midrange_jax(
    states: jnp.ndarray,
    state_prior: jnp.ndarray,
    k: float,
    q_low: float = 0.2,
    q_high: float = 0.8,
) -> float:
    """
    Compute the context-dependent mid-range max-min threshold θ_k(C).

    θ_k(C) = x_max^mid(C) - k * (x_max^mid(C) - x_min^mid(C))

    Args:
        states:      (n_obj, 3)
        state_prior: (n_obj,)
        k:           interpolation parameter in [0, 1]
        q_low:       lower quantile for mid-range band
        q_high:      upper quantile for mid-range band

    Returns:
        theta_k: scalar threshold.
    """
    x_min_mid, x_max_mid = get_midrange_extrema_from_context(
        states, state_prior, q_low=q_low, q_high=q_high
    )
    theta_k = x_max_mid - k * (x_max_mid - x_min_mid)
    return theta_k

def size_semantic_value_midrange(
    size: jnp.ndarray,      # scalar or array
    theta_k: float,
    w_f: float,
) -> jnp.ndarray:
    """
    Graded size semantics with mid-range threshold θ_k(C) and noise w_f.

    ⟦size⟧_C(s) = 1 - Φ( (x - θ_k) / (w_f * sqrt(x^2 + θ_k^2)) )
    """
    # Avoid zero division; eps not strictly necessary but helps numerics.
    eps = 1e-8
    denom = w_f * jnp.sqrt(size**2 + theta_k**2 + eps)
    z = (size - theta_k) / denom
    # Standard normal CDF at z
    return 1.0 - dist.Normal(0.0, 1.0).cdf(z)

def _size_meaning_vector(
    states: jnp.ndarray,   # (n_obj, 3)
    prior: jnp.ndarray,    # (n_obj,)
    k: float,
    wf: float,
    q_low: float = 0.2,
    q_high: float = 0.8,
) -> jnp.ndarray:
    """Context-dependent graded size semantics ⟦size⟧_C(s) for all objects."""
    sizes = states[:, 0]  # (n_obj,)

    # mid-range extrema & threshold θ_k(C)
    x_min_mid, x_max_mid = get_midrange_extrema_from_context(
        states, prior, q_low=q_low, q_high=q_high
    )
    theta_k = x_max_mid - k * (x_max_mid - x_min_mid)

    # ⟦size⟧_C(s) = 1 - Φ((x - θ_k)/(wf * sqrt(x²+θ_k²)))
    eps = 1e-8
    denom = wf * jnp.sqrt(sizes**2 + theta_k**2 + eps)
    z = (sizes - theta_k) / denom
    return 1.0 - dist.Normal(0.0, 1.0).cdf(z)  # (n_obj,)

# --- atomic adjective meaning (no sampling, just semantic values) ---

def adjMeaning(
    word: int,                 # 0 = size (“big”), 1 = color (“blue”)
    states: jnp.ndarray,       # (n_obj, 3)
    current_state_prior: jnp.ndarray,  # (n_obj,)
    color_semvalue: float = 0.8,
    wf: float = 0.6,
    k: float = 0.5,
) -> jnp.ndarray:
    """
    Deterministic lexical meaning ⟦w⟧_C(s) for one adjective w in context C.

    Returns a vector of shape (n_obj,) with semantic values in [0,1].
    Uses JAX control flow (lax.cond), so it is vmap/jit-safe.
    """

    def color_branch(_):
        # Bernoulli-style color semantics, context independent
        x_color = states[:, 1]  # 0/1 feature
        return x_color * color_semvalue + (1.0 - x_color) * (1.0 - color_semvalue)

    def size_branch(_):
        # Context-dependent graded size semantics
        return size_semantic_value_midrange(
            size=states[:, 0],
            theta_k=get_threshold_k_midrange_jax(
                states=states,
                state_prior=current_state_prior,
                k=k,
            ),
            w_f=wf,
        )

    # word == 1 → color, else → size (we only ever use 0 or 1 here)
    return lax.cond(word == 1, color_branch, size_branch, operand=None)

def literal_listener_one_word(
    states: jnp.ndarray,              # (n_obj, 3)
    color_semvalue: float = 0.90,
    wf: float = 1,
    k: float = 0.5,
) -> jnp.ndarray:
    """
    Incremental literal listener for single-word utterances:
      row 0: L(s | 'big')
      row 1: L(s | 'blue')
    Returns array of shape (2, n_obj).
    """
    n_obj = states.shape[0]
    prior0 = jnp.ones(n_obj) / n_obj  # uniform prior over states

    # tokens: 0 = size ("big"), 1 = color ("blue")
    tokens = jnp.array([0, 1], dtype=jnp.int32)

    def update_token(token: int) -> jnp.ndarray:
        m = adjMeaning(
            word=token,
            states=states,
            current_state_prior=prior0,
            color_semvalue=color_semvalue,
            wf=wf,
            k=k,
        )
        unnorm = prior0 * m
        norm = jnp.clip(unnorm.sum(), 1e-20)
        return unnorm / norm  # (n_obj,)

    # shape: (2, n_obj)
    return jax.vmap(update_token)(tokens)

def literal_listener_recursive(
    word_length: int,
    states: jnp.ndarray,              # (n_obj, 3)
    color_semvalue: float = 0.90,
    wf: float = 1,
    k: float = 0.5,
) -> jnp.ndarray:
    """
    Incremental literal listener using backward functional semantics.

    We only need two utterances:
      - row 0: "big blue"  (size → color)   encoded as [0, 1]
      - row 1: "blue big"  (color → size)   encoded as [1, 0]

    Returns:
        probs: jnp.ndarray of shape (2, n_obj)
               row 0: P(s | "big blue")
               row 1: P(s | "blue big")
    """
    assert word_length == 2, "This implementation is tailored to 2-word utterances."

    n_obj = states.shape[0]
    # Prior over states P(s): uniform
    prior0 = jnp.ones(n_obj) / n_obj

    # encode the two utterances: 0=size, 1=color
    utterances = jnp.array([
        [0, 1],  # "big blue"
        [1, 0],  # "blue big"
    ], dtype=jnp.int32)

    def update_one_token(prior: jnp.ndarray, token: int) -> jnp.ndarray:
        """
        Single backward update:
          P_{t-1}(s) ∝ P_t(s) * ⟦w_t⟧_{C_t}(s),
        where C_t is encoded by the current prior.
        """
        m = adjMeaning(
            word=token,
            states=states,
            current_state_prior=prior,
            color_semvalue=color_semvalue,
            wf=wf,
            k=k,
        )
        unnorm = prior * m
        norm = jnp.clip(unnorm.sum(), 1e-20)
        return unnorm / norm

    def interpret_utterance(tokens: jnp.ndarray) -> jnp.ndarray:
        """
        Backward composition for a single utterance u = (w1, w2):
          start with P_2(s) = P_prior(s),
          then apply w_2, then w_1.
        """
        tokens_rev = tokens[::-1]  # right-to-left
        def step(p, tok):
            new_p = update_one_token(p, tok)
            return new_p, None

        final_belief, _ = lax.scan(step, prior0, tokens_rev)
        return final_belief  # (n_obj,)

    # vmap over the 2 utterances → (2, n_obj)
    probs = jax.vmap(interpret_utterance)(utterances)
    return probs

def speaker_recursive(
    word_length,
    states: jnp.ndarray,
    alpha: float = 1.0,
    bias: float = 2.0,
    color_semvalue: float = 0.98,
    form_semvalue: float = 0.98,  # kept for API compatibility, unused here
    wf: float = 0.6,
    k: float = 0.5,
):
    """
    Truly token-wise incremental RSA speaker for the 2-utterance slider setup.

    Utterances:
      u_0 = "big blue"  (size -> color)
      u_1 = "blue big"  (color -> size)

    Returns:
        S_inc: jnp.ndarray of shape (n_obj, 2)
               S_inc[j, 0] = P_inc("big blue" | s_j)
               S_inc[j, 1] = P_inc("blue big" | s_j)
    """
    assert word_length == 2, "speaker_recursive is currently implemented for 2-word utterances."

    n_obj = states.shape[0]

    # --- 1. Literal listeners for 1-word and 2-word utterances ---

    # L1: shape (2, n_obj)
    #   row 0: L(s | 'big')
    #   row 1: L(s | 'blue')
    L1 = literal_listener_one_word(
        states,
        color_semvalue=color_semvalue,
        wf=wf,
        k=k,
    )  # (2, n_obj)

    # L2: shape (2, n_obj)
    #   row 0: L(s | 'big blue')
    #   row 1: L(s | 'blue big')
    L2 = literal_listener_recursive(
        word_length,
        states,
        color_semvalue=color_semvalue,
        wf=wf,
        k=k,
    )  # (2, n_obj)

    # transpose to (n_obj, 2) for easier per-state handling
    L1_T = L1.T  # (n_obj, 2)
    L2_T = L2.T  # (n_obj, 2)

    # unpack for readability
    L1_big  = L1_T[:, 0]   # L(s_j | 'big')
    L1_blue = L1_T[:, 1]   # L(s_j | 'blue')

    L2_bigblue = L2_T[:, 0]  # L(s_j | 'big blue')
    L2_bluebig = L2_T[:, 1]  # L(s_j | 'blue big')

    eps = 1e-20

    # --- 2. Step 1: choose first token (big vs blue) ---

    num_big  = jnp.power(jnp.clip(L1_big,  eps, 1.0), alpha)
    num_blue = jnp.power(jnp.clip(L1_blue, eps, 1.0), alpha)
    Z1 = jnp.clip(num_big + num_blue, eps)

    S1_big  = num_big  / Z1   # P_1(w1='big'  | s_j)
    S1_blue = num_blue / Z1   # P_1(w1='blue' | s_j)

    # --- 3. Step 2: given prefix, choose STOP or the second adjective ---

    # Prefix 'big': next is STOP (-> 'big') or 'blue' (-> 'big blue')
    num_bigblue = jnp.power(jnp.clip(L2_bigblue, eps, 1.0), alpha)  # candidate second word 'blue'
    num_stop_big = jnp.power(jnp.clip(L1_big, eps, 1.0), alpha)     # stopping after 'big'
    Z2_big = jnp.clip(num_bigblue + num_stop_big, eps)

    S2_blue_given_big = num_bigblue / Z2_big   # P_2(w2='blue' | prefix='big', s_j)

    # Prefix 'blue': next is STOP (-> 'blue') or 'big' (-> 'blue big')
    num_bluebig = jnp.power(jnp.clip(L2_bluebig, eps, 1.0), alpha)  # candidate second word 'big'
    num_stop_blue = jnp.power(jnp.clip(L1_blue, eps, 1.0), alpha)   # stopping after 'blue'
    Z2_blue = jnp.clip(num_bluebig + num_stop_blue, eps)

    S2_big_given_blue = num_bluebig / Z2_blue  # P_2(w2='big' | prefix='blue', s_j)

    # --- 4. Path probabilities for length-2 utterances (chain rule) ---

    # P_chain("big blue" | s_j) ∝ S1(big|s_j) * S2(blue | prefix='big', s_j)
    P_chain_bigblue = S1_big * S2_blue_given_big

    # P_chain("blue big" | s_j) ∝ S1(blue|s_j) * S2(big | prefix='blue', s_j)
    P_chain_bluebig = S1_blue * S2_big_given_blue

    # stack: (n_obj, 2)
    P_chain = jnp.stack([P_chain_bigblue, P_chain_bluebig], axis=-1)
    P_chain = jnp.clip(P_chain, eps, 1.0)

    # --- 5. Add utterance-level cost (bias against "blue big") ---

    # cost[0] = 0 for "big blue"
    # cost[1] = bias for "blue big"
    utt_cost = jnp.array([0.0, bias])  # (2,)

    # Utilities: log P_chain(u | s_j) - cost(u)
    logP_chain = jnp.log(P_chain)
    util = logP_chain - utt_cost  # broadcast over states

    # --- 6. Final incremental speaker distribution over the 2 utterances ---

    S_inc = jax.nn.softmax(util, axis=-1)  # (n_obj, 2)

    # Extract only the probs for "big blue" and target referent (index 0)
    probs_bigblue_referent = S_inc[0, 0]  # (n_obj,)
    return probs_bigblue_referent

def global_speaker(
    states,
    alpha: float = 1.0,
    bias: float = 2.0,
    color_semvalue: float = 0.98,
    k: float = 0.5,
):
    listener = literal_listener_recursive(
        2,
        states,
        color_semvalue=color_semvalue,
        k=k,
    )  # (2, n_obj)

    utt_cost = jnp.array([0.0, bias])          # (2,)
    util_speaker = jnp.log(listener.T) - utt_cost  # (n_obj, 2)
    softmax_result = jax.nn.softmax(alpha * util_speaker, axis=-1)

    # Extract only the probs for "big blue" and target referent (index 0)
    probs_bigblue_referent = softmax_result[0, 0]  # (n_obj,)
    return probs_bigblue_referent

vectorized_gb_speaker = jax.vmap(global_speaker, in_axes=(0, # states
                                                          None, # alpha
                                                          None, # bias
                                                          None, # color_semvalue
                                                          None, # k
                                                          ))

# Define a function to encode the states of the objects
def encode_states(line):
      states = []
      for i in range(6):
        color = 1 if line.iloc[10 + i] == "blue" else 0
        form = 1 if line.iloc[16 + i] == "circle" else 0
        new_obj = (line.iloc[4 + i], color, form) # size, color, form
        states.append(new_obj)
      return jnp.array(states)

def import_dataset(file_path = "../01-dataset/01-slider-data-preprocessed.csv"):
   # Import the data
    df = pd.read_csv(file_path)

    # Subset data to only include combination dimension_color
    df = df[df['combination'] == 'dimension_color']
    df.reset_index(inplace=True, drop=True)

    # Mutate the dataset to include the states of the objects
    df["states"] = df.apply(lambda row: encode_states(row), axis=1)

    # Transform/rescale slider value from range 0 to 100 to 0 to 1
    df.prefer_first_1st = jnp.clip(df.prefer_first_1st.to_numpy(), 0, 100)
    df.prefer_first_1st = df.prefer_first_1st/100

        
    # split the dataset into training and test sets
    #train, test = train_test_split(df, test_size=0.99, random_state=42)

    # Use the whole dataset as training set
    train = df

    # Extract the states and empirical data and store them in JAX arrays
    states_train = jnp.stack([cell for cell in train.states])
    empirical_train = jnp.array(train.prefer_first_1st.to_numpy())

    return states_train, empirical_train, df
    

def test_core_rsa():
    # Import the data
    file_path = "../01-dataset/01-slider-data-preprocessed.csv"
    df = pd.read_csv(file_path)

    # subset data to only include combination dimension_color
    df = df[df['combination'] == 'dimension_color']
    df.reset_index(inplace=True, drop=True)

    # Mutate the dataset to include the states of the objects
    df_experiment = df.copy()
    df_experiment["states"] = df_experiment.apply(lambda row: encode_states(row), axis=1)

    # Transform/rescale slider value from range 0 to 100 to 0 to 1
    df_experiment.prefer_first_1st = jnp.clip(df_experiment.prefer_first_1st.to_numpy(), 0, 100)
    df_experiment.prefer_first_1st = df_experiment.prefer_first_1st/100
    print(df_experiment.prefer_first_1st.describe())

    # 118 brdc sharp, target not the biggest
    # 12 brdc blurred

    # 119 erdc sharp
    # 15 erdc blurred

    index = 119
    states_manuell = jnp.array([[10., 1., 1.],
                    [3., 1., 1.],
                    [3., 1., 1.],
                    [3., 1., 0.],
                    [3., 1., 0.],
                    [1., 0., 1.]], dtype=jnp.float32)

    #states_example = states_manuell
    states_example = df_experiment.iloc[index, df_experiment.columns.get_loc("states")]
    condition = df_experiment.iloc[index, df_experiment.columns.get_loc("conditions")]
    distribution = df_experiment.iloc[index, df_experiment.columns.get_loc("sharpness")]
    preference = df_experiment.iloc[index, df_experiment.columns.get_loc("prefer_first_1st")]
    print(states_example)
    print(condition + " " + distribution)
    print(preference)
    print(f"literal listener one word: {literal_listener_one_word(states_example, color_semvalue=0.8)}")
    print(f"literal listener two words: {literal_listener_recursive(2,states_example, color_semvalue=0.8)}")
    print(f"speaker one word: {global_speaker(states_example, color_semvalue=0.8)}")
    print("________________________________________")
    print(f"speaker two words incremental: {speaker_recursive(2,states_example, color_semvalue=0.8)}")


def link_logit(p,s):
    x0 = 1 / (jnp.exp(1 / (-2 * s)) + 1)
    xtrans = p * (x0 - (1 - x0)) + (1 - x0)
    return s * -jnp.log((1 / xtrans) - 1) + 0.5

vectorized_inc_speaker = jax.vmap(speaker_recursive, in_axes=(None,0,None,None,None,None,None,None))

def likelihood_gb_speaker(states = None, data = None):
    alpha = numpyro.sample("alpha", dist.HalfNormal(5))
    color_semvalue = numpyro.sample("color_semvalue", dist.Beta(8, 3))
    k = numpyro.sample("k", dist.Beta(3, 8))
    bias = numpyro.sample("bias", dist.HalfNormal(5))
    sigma = numpyro.sample("sigma", dist.HalfNormal(0.1))

    with numpyro.plate("data",len(states)):
        model_prob = vectorized_gb_speaker(states, 
                                           alpha, 
                                           bias, 
                                           color_semvalue,
                                            k)
        #slider_predict = jax.vmap(link_logit, in_axes = (0,None))(model_prob[:,0,0], steepness)
        #slider_predict = jnp.clip(slider_predict, 1e-5, 1 - 1e-5)
        if data is not None:
            data = jnp.clip(data, 1e-5, 1 - 1e-5)
        numpyro.sample("obs", dist.TruncatedNormal(model_prob, sigma, low = 1e-5, high = 1 - 1e-5,), obs=data)

def likelihood_inc_speaker(states = None, data = None):
    gamma = numpyro.sample("gamma", dist.HalfNormal(5))
    color_semvalue = numpyro.sample("color_semvalue", dist.Uniform(0.5, 1))
    form_semvalue = color_semvalue
    k = numpyro.sample("k", dist.Uniform(0, 1))
    wf = 0.8
    bias = numpyro.sample("bias", dist.HalfNormal(5))
    steepness = 1
    sigma = numpyro.sample("sigma", dist.HalfNormal(0.1))

    with numpyro.plate("data",len(states)):
        model_prob = vectorized_inc_speaker(2, states, gamma, bias, color_semvalue, form_semvalue, wf, k)
        #slider_predict = jax.vmap(link_logit, in_axes = (0,None))(model_prob[:,0,0], steepness)
        #slider_predict = jnp.clip(slider_predict, 1e-5, 1 - 1e-5)
        if data is not None:
            data = jnp.clip(data, 1e-5, 1 - 1e-5)
        numpyro.sample("obs", dist.Normal(model_prob, sigma), obs=data)

def run_inference(
    speaker_type: str = "global",
    num_samples: int = 1000,
    num_warmup: int = 1000,
    num_chains: int = 4,

):
    # Import the dataset
    states_train, empirical_train, df = import_dataset()

    # Setup output file name
    output_file_name = f"./inference_data/mcmc_results_{speaker_type}_speaker_warmup{num_warmup}_samples{num_samples}_chains{num_chains}.nc"

    # Define a random key
    rng_key = random.PRNGKey(4711)
    rng_key, rng_key_ = random.split(rng_key)

    # Define the MCMC kernel
    rng_key = random.PRNGKey(11)
    rng_key, rng_key_ = random.split(rng_key)
    if speaker_type == "global":
        model = likelihood_gb_speaker
    elif speaker_type == "incremental":
        model = likelihood_inc_speaker
    else:
        raise ValueError("Invalid speaker type. Choose 'global' or 'incremental'.")
    
    # Define and run MCMC
    kernel = NUTS(model, dense_mass=True, max_tree_depth=10, target_accept_prob=0.9)
    mcmc = MCMC(kernel, num_warmup=num_warmup,num_samples=num_samples,num_chains=num_chains)
    mcmc.run(rng_key_, states_train, empirical_train)

    # Print the summary of the posterior distribution
    mcmc.print_summary()

    # Get the MCMC samples and convert to a numpyro ArviZ InferenceData object
    posterior_samples = mcmc.get_samples() 
    posterior_predictive = Predictive(model, posterior_samples)(
    PRNGKey(1), states_train
    )
    prior = Predictive(model, num_samples=1000)(
        PRNGKey(2), states_train
    )   

    N = states_train.shape[0]  # 3196

    numpyro_data = az.from_numpyro(
        mcmc,
        prior=prior,
        posterior_predictive=posterior_predictive,
        coords={"states": np.arange(N)},
        dims={"obs": ["states"]},
    )

    # Write the inference data to a netcdf file
    az.to_netcdf(numpyro_data, output_file_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run speaker inference with NumPyro.")
    parser.add_argument("--speaker_type", type=str, choices=["global", "incremental"], default="global",
                        help="Choose the speaker model type.")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of posterior samples.")
    parser.add_argument("--num_warmup", type=int, default=1000, help="Number of warm-up iterations.")
    parser.add_argument("--num_chains", type=int, default=4, help="Number of MCMC chains.")
    parser.add_argument("--test", action="store_true", help="Run test function and exit.")


    args = parser.parse_args()

    numpyro.set_platform("cpu")
    numpyro.set_host_device_count(args.num_chains)
    jax.local_device_count()

    if args.test:
        pass
    else:
        run_inference(
            speaker_type=args.speaker_type,
            num_samples=args.num_samples,
            num_warmup=args.num_warmup,
            num_chains=args.num_chains,
        )