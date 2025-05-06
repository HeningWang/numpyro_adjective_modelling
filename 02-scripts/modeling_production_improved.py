import pandas as pd
import numpy as np
import jax.numpy as jnp
from typing import Tuple, Dict, List, Any, Sequence, Callable
from jax import jit, vmap
import jax
from jax import lax
from jax import random
from jax.random import PRNGKey, split
from functools import partial

import numpyro
from numpyro.diagnostics import hpdi
import numpyro.distributions as dist
from numpyro import handlers
from numpyro.infer import MCMC, NUTS, HMC, MixedHMC
from numpyro.infer import Predictive
from sklearn.model_selection import train_test_split
numpyro.set_platform("cpu")

print(jax.__version__)
jax.devices()

# ========================
# Global Variables (placeholders)
# ========================
utterance_list = None
utterance_prior = None

def normalize(arr, axis=1):
    """
    Normalize arr along axis
    """
    return arr / jnp.sum(arr, axis=axis, keepdims=True)

def import_dataset(
    file_path: str = "../01-dataset/01-production-data-preprocessed.csv"
    ) -> Tuple[jnp.ndarray, jnp.ndarray, pd.DataFrame]:
    """
    Reads a CSV of production data, encodes object states and utterances.

    Returns:
      - states_train: jnp.ndarray of shape (N, 6, 3) with 6 objects that have 3 attributes (size, color, form)
      - empirical_train: jnp.ndarray of shape (N,) with integer utterance codes
      - df: the original DataFrame with two new columns:
          * 'states' as (6,3) arrays
          * 'annotation_encoded' as integer codes
    """
    # 1) Load & drop missing
    df = pd.read_csv(file_path)
    df = df.dropna(subset=["annotation"]).copy()
    # Subset the data by conditions to include only the relevant ones
    df = df[df["conditions"].isin(["erdc", "erdf"])]

    # 2) Encode states via vectorized slicing
    #    – sizes are numeric already
    #    – colors: "blue"→1, else 0
    #    – forms:  "circle"→1, else 0
    sizes = df.iloc[:, 6:12].to_numpy(dtype=float)                  # (N,6)
    colors = (df.iloc[:, 12:18] == "blue").to_numpy(dtype=int)      # (N,6)
    forms  = (df.iloc[:, 18:24] == "circle").to_numpy(dtype=int)    # (N,6)

    # Stack into shape (N,6,3)
    states_np = np.stack([sizes, colors, forms], axis=2) 
    states_train = jnp.array(states_np)                # (N,6,3)

    # 3) flat codes for utterances
    # Make it categorical from strings to integers, each unique string is a category
    df["annotation"] = df["annotation"].astype("category")
    cats = list(df["annotation"].cat.categories)         # e.g. ["D","C","CD",...]
    
    # Build maps
    utt2idx: Dict[str,int] = {u:i for i,u in enumerate(cats)}
    idx2utt: Dict[int,str] = {i:u for u,i in enumerate(cats)}
    
    # Stack into shape (N,)
    empirical_flat = jnp.array(df["annotation"].cat.codes.to_numpy(), dtype=jnp.int32) # (N,)
    

    # 4) Build a sequence of symbols for each utterance
    symbol2idx = {"D": 0, "C": 1, "F": 2}
    max_len = 3

    # Pull out the raw strings in plain Python
    utt_strings = df["annotation"].tolist()   # e.g. ["D","CF","FDC",...]

    # Turn each string into a list of symbol‐ints
    sequences = []
    for u in utt_strings:
        seq = [symbol2idx[ch] for ch in u]
        # pad with -1 up to max_len
        seq = seq + [-1] * (max_len - len(seq))
        sequences.append(seq)

    # 4) Stack into one jnp.ndarray
    empirical_seq = jnp.array(sequences, dtype=jnp.int32)  # shape (N, max_len)

    # Get the unique utterances from empirical_seq
    unique_utterances = jnp.unique(empirical_seq, axis=0)  # shape (U, max_len)

    # 5) Build a mask array to ignore padding
    seq_mask = jnp.array(
        [[True]*len(u) + [False]*(max_len - len(u)) for u in utt_strings],
        dtype=bool
    )  # shape (N, max_len)

    # 6) Build a flat codes for the utterances given the sequence
    # Build a dict mapping each unique utterance to its index
    uttSeq2idx = {tuple(u): i for i, u in enumerate(unique_utterances.tolist())}
    # Map each utterance in the sequence to its index
    empirical_seq_flat = jnp.array(
        [uttSeq2idx[tuple(u)] for u in empirical_seq.tolist()],
        dtype=jnp.int32
    )  # shape (N,)

    # 7) Add to the DataFrame
    df["statesArray"] = states_train.tolist()  # shape (N,)
    df["annotation_string_flat"] = empirical_flat.tolist()  # shape (N,)
    df["annotation_seq"] = empirical_seq.tolist()  # shape (N,)
    df["annotation_seq_mask"] = seq_mask.tolist()  # shape (N, max_len)
    df["annotation_seq_flat"] = empirical_seq_flat.tolist()  # shape (N,)

    return states_train, empirical_flat, empirical_seq, seq_mask, df, unique_utterances, empirical_seq_flat

def build_utterance_prior(utterance_list: List[str],
                        costParam_length: float = 1.0,
                        costParam_bias: float = 1.0,
                        costParam_subjectivity: float = 1.0
                          ) -> jnp.ndarray:
    """
    Given a list of utterance categories (e.g. ["D","C","F","CD",...]),
    returns a (U,) prior over them, where:
      - length 1 → util 3; length 2 → util 2; length 3 → util 1
      - utterances in {'CD','FD','CFD','FCD','FDC'} get a -0.5 penalty
      - utterances starting with 'D' get a +1.0 boost
    Finally, utilities are softmaxed into probabilities.
    """
    # 1) Base utils by length
    base_utils = jnp.array([3 if len(u)==1 else 2 if len(u)==2 else 1
                            for u in utterance_list], dtype=jnp.float32)

    # 2) Penalty for specific utterances
    penalized = jnp.array([
        [1, 0, -1], #"CD"
        [2, 1, -1], # "FD"
        [1, 2, 0],  # "CFD"
        [2, 1, 0],  # "FCD"
        [2, 0, 1]   # "FDC"
    ])

    def is_penalized(u, penalized):
    # u has shape (3,)
    # penalized_seq has shape (5,3)
        match = jnp.all(u == penalized, axis=1)  # shape (5,), True where matches
        return jnp.any(match)  # True if any row matches

    # Assume utterance_list is a list or array of (N,3)
    penalty = jnp.array([
        -0.5 if is_penalized(u, penalized) else 0.0
        for u in utterance_list
    ], dtype=jnp.float32)

    # penalty = jnp.array([ -0.5 if u in penalized_seq else 0.0
    #                       for u in utterance_list], dtype=jnp.float32)

    # 3) Boost for utterances starting with 'D'
    boost = jnp.array([ 1.0 if u[0] == 0 else 0.0
                        for u in utterance_list], dtype=jnp.float32)

    # 4) Combine utilities
    utils = costParam_length * base_utils + costParam_bias * penalty + costParam_subjectivity * boost    # shape (U,)

    # 5) Softmax into a proper prior
    return jax.nn.softmax(utils, axis=0)

# ========================
# Global Variables (Setup)
# ========================
utterance_list = import_dataset()[5]
utterance_prior = build_utterance_prior(utterance_list)


def size_semantic_value(
    size: float,
    threshold: float,
    wf: float = 0.5
) -> float:
    """
    P(size adjective applies | size, threshold, wf), using a Normal slack.
    
    Args:
        size: the numeric size of the object.
        threshold: the cutoff such that sizes above it are more likely 'big'.
        wf: weight factor controlling the standard deviation.
        
    Returns:
        A probability in [0,1].
    """
    mu = size - threshold
    sigma = wf * jnp.sqrt(size**2 + threshold**2)
    return 1.0 - dist.Normal(mu, sigma).cdf(0.0)


def get_threshold_kp_sample_jax(states, states_prior, k=0.5):
    """
    Samples half the objects according to state_prior, then returns
    a k‐weighted threshold between their min & max sizes.
    
    Args:
        sizes: array of object sizes, shape (n_objs,).
        state_prior: categorical probs over objects, shape (n_objs,).
        rng_key: PRNGKey for randomness.
        k: controls interpolation: 0→max, 1→min.
        
    Returns:
        threshold: the interpolated cutoff.
        new_key: updated RNG key.
    """
    sample_size = int(round(states.shape[0] / 2)) # Sample size is half of the number of objects in a given context
    costum_dist = dist.Categorical(probs=states_prior)
    sample_indices = jnp.unique(costum_dist.sample(random.PRNGKey(0),(1,sample_size)), size= sample_size)
    sorted_states = states[sample_indices][:,0]
    min_val = jnp.min(sorted_states)
    max_val = jnp.max(sorted_states)

    weighted_threshold = max_val - k * (max_val - min_val)
    return weighted_threshold


def meaning(
    word: str,
    states: jnp.ndarray,
    state_prior: jnp.ndarray,
    color_semval: float = 0.95,
    form_semval:  float = 0.95,
    k: float = 0.5,
    wf: float = 0.5
) -> Tuple[jnp.ndarray, PRNGKey]:
    """
    Literal semantics P(obj | word) for a single-word utterance.
    
    Args:
        word: one of "D","C","F" encoded in integers 0,1,2.
        0: "size", 1: "color", 2: "form", -1: "padding".
        states: array (n_objs, 3) with columns [size, color_bit, form_bit].
        state_prior: uniform (or other) prior over objects.
        rng_key: PRNGKey for any randomness (threshold sampling).
        color_semval: P('color' applies | color_bit=1).
        form_semval:  P('form' applies | form_bit=1).
        k, wf: parameters for size semantics.
        
    Returns:
        probs: jnp.ndarray shape (n_objs,) summing to 1.
        rng_key: updated key.
    """
    sizes  = states[:, 0]
    colors = states[:, 1]
    forms  = states[:, 2]
    
    if word == 1: # "color"
        raw = jnp.where(colors == 1, color_semval, 1.0 - color_semval)

    if word == 2: # "form"
        raw = jnp.where(forms == 1, form_semval, 1.0 - form_semval)

    if word == 0: # "size"
        threshold = get_threshold_kp_sample_jax(states, state_prior, k)
        raw = jax.vmap(size_semantic_value, in_axes = (0, None, None))(states, threshold, wf)[:,0] # Apply the meaning function for size adjective

    if word == -1: # "padding"
        raw = jnp.ones(states.shape[0], dtype=jnp.float32)

    # normalize into a probability distribution
    probs = normalize(raw, axis=0)
    return probs

def meaning_jax(
    word: int,
    states: jnp.ndarray,
    state_prior: jnp.ndarray,
    color_semval: float,
    form_semval: float,
    k: float,
    wf: float,
) -> jnp.ndarray:
    
    sizes = states[:, 0]
    colors = states[:, 1]
    forms  = states[:, 2]

    def size_case(_):
        threshold = get_threshold_kp_sample_jax(states, state_prior, k)
        return jax.vmap(size_semantic_value, in_axes=(0, None, None))(states, threshold, wf)[:, 0]

    def color_case(_):
        return jnp.where(colors == 1, color_semval, 1.0 - color_semval)

    def form_case(_):
        return jnp.where(forms == 1, form_semval, 1.0 - form_semval)

    def padding_case(_):
        return jnp.ones(states.shape[0], dtype=jnp.float32)

    # remap -1 (padding) to 3
    index = jnp.where(word == -1, 3, word)

    raw = lax.switch(
        index,
        [size_case, color_case, form_case, padding_case],
        operand=None
    )

    return normalize(raw, axis=0)

def incremental_semantics(
    states:       jnp.ndarray,   # shape (n_objs,3)
    color_sem:    float = 0.95,
    form_sem:     float = 0.95,
    k:            float = 0.5,
    wf:           float = 0.5,
) -> jnp.ndarray:
    """
    Compute P(obj | utterance tokens u_tokens) by:
     - for each token != -1, call meaning(...)
     - multiply their semantic vectors
     - normalize to sum=1

    Args:
        utterances: array of utterance tokens, shape (n_utt,).
        states: array of object states, shape (n_objs,3).
        color_sem: P('color' applies | color_bit=1).
        form_sem:  P('form' applies | form_bit=1).
        k, wf: parameters for size semantics.

    Returns:
        joint: jnp.ndarray of shape (n_utt, n_obj) summing to 1.
    """
    # 1) Implement the logic for get incremental semantics of single utterance
    # For single utterance, we apply the meaning function to each token and multiply the results together
    # Crucially, the results of computation depend on the order of the tokens, so it should be done in a recursive way, starting from the last token

    # 2) Apply the incremental semantics to all other utterances given one state

    # 3) Stack the results into a matrix
    # 4) Normalize the result to sum to 1, row-wise
    # 5) Return the joint distribution
    utterances = utterance_list
    n_objs = states.shape[0]
    M = []
    state_prior = jnp.ones(n_objs) / n_objs  # uniform prior over objects

    # Iterate over each utterance
    for utt in utterances:
        prior = state_prior
        # Iterate from last real token to first
        for token in reversed(utt):
            # if token < 0:
            #     continue  # skip padding
            posterior = meaning(
                token, states, prior,
                color_sem, form_sem, k, wf
            )
            # normalize just in case
            prior = posterior
        M.append(prior)
    
    # Stack into a matrix
    M = jnp.stack(M, axis=0)  # shape (U, n_objs)

    return M

def incremental_semantics_jax(
    states:       jnp.ndarray,   # (n_obj, 3)
    color_sem:    float = 0.95,
    form_sem:     float = 0.95,
    k:            float = 0.5,
    wf:           float = 0.5,
    state_prior: jnp.ndarray = None
) -> jnp.ndarray:
    """
    Compute P(obj | utterance) for all utterances using backward functional semantics.

    Returns:
        M: shape (n_utt, n_obj)
    """
    utterances = utterance_list  # assumed global
    n_obj = states.shape[0]
    if state_prior is None:
        state_prior = jnp.ones(n_obj) / n_obj  # uniform prior

    def apply_tokens(tokens: jnp.ndarray) -> jnp.ndarray:
        def step(prior, token):
            def skip(_):
                return prior
            def apply(_):
                return meaning_jax(token, states, prior, color_sem, form_sem, k, wf)
            posterior = lax.cond(token < 0, skip, apply, operand=None)
            return posterior, None

        final_belief, _ = lax.scan(step, state_prior, tokens[::-1])
        return final_belief

    # Apply over all utterances using vmap
    M = jax.vmap(apply_tokens)(utterances)  # shape (n_utt, n_obj)
    return M

def global_speaker(
    states: jnp.ndarray,               # shape (n_objs,3)
    alpha: float = 1.0,
    color_semval: float = 0.95,
    k: float = 0.5,
    utterance_prior: jnp.ndarray = None
):
    """
    Output: probs for each state
    """
    current_utt_prior = jnp.log(utterance_prior)
    meaning_matrix = incremental_semantics_jax(states, color_semval, k)
    util_speaker = jnp.log(jnp.transpose(meaning_matrix)) + current_utt_prior
    softmax_result = jax.nn.softmax(alpha * util_speaker)
    return softmax_result

def incremental_speaker(
    states: jnp.ndarray,
    alpha: float = 1.0,
    color_semval: float = 0.95,
    k: float = 0.5,
    utterance_prior: jnp.ndarray = None
):
    """
    Output: probs for each state
    """
    utterances = utterance_list
    n_objs = states.shape[0]
    M = []
    # state_prior = jnp.ones(n_objs) / n_objs  # uniform prior over objects
    
    # meaning_matrix = incremental_semantics_jax(states, color_semval, k)
    # util_speaker = jnp.log(jnp.transpose(meaning_matrix)) + utt_prior
    # softmax_result = jax.nn.softmax(alpha * util_speaker)
    # utt_posterior = softmax_result
    if utterance_prior is None:
        utt_prior = jnp.log(utterance_prior)
    def apply_tokens(tokens: jnp.ndarray) -> jnp.ndarray:
        """
        Given a single utterance (shape (3,)), compute posterior distribution over objects.
        Applies meaning(...) recursively from right to left using lax.scan.
        """

        def step(utt_prior, token):
            # Skip padding (-1): return unchanged prior
            def skip(_):
                return utt_prior

            def apply(_):
                return global_speaker(states, alpha, color_semval, k)[0,:] # Get the probs of utterances given the first state, referent is always the first state

            posterior = lax.cond(token < 0, skip, apply, operand=None)
            utt_prior = jnp.log(posterior)
            return posterior, None  # carry only prior forward
        
        # Loop over the tokens in conventional order (left to right)
        final_prior, _ = lax.scan(step, utt_prior, tokens[::1])
        return final_prior
    
    # Apply over all utterances using vmap
    M = jax.vmap(apply_tokens)(utterances)  # shape (n_utt, n_obj)
    return M

def likelihood_function_global_speaker(states = None, empirical = None):
    alpha = numpyro.sample("gamma", dist.HalfNormal(5))
    # alpha = 1
    color_semval = numpyro.sample("color_semvalue", dist.Uniform(0, 1))
    #color_semval = 0.8
    #k = numpyro.sample("k", dist.Uniform(0, 1))
    k = 0.5
    utt_probs_conditionedReferent = global_speaker(states, alpha, color_semval, k)[0,:] # Get the probs of utterances given the first state, referent is always the first state
    with numpyro.plate("data", len(states)):
        if empirical is None:
            numpyro.sample("obs", dist.Categorical(probs=utt_probs_conditionedReferent))
        else:
            numpyro.sample("obs", dist.Categorical(probs=utt_probs_conditionedReferent), obs=empirical)

def run_inference():
    states_train, empirical_train_flat, empirical_train_seq, _, _, _ = import_dataset()
    print("States train shape:", states_train.shape)
    print("Empirical train shape:", empirical_train_seq.shape)
    print("Empirical train flat shape:", empirical_train_flat.shape)
    # define the MCMC kernel and the number of samples
    rng_key = random.PRNGKey(11)
    rng_key, rng_key_ = random.split(rng_key)

    kernel = NUTS(likelihood_function_global_speaker)
    #kernel = MixedHMC(HMC(likelihood_function, trajectory_length=1.2), num_discrete_updates=20)
    mcmc_inc = MCMC(kernel, num_warmup=100,num_samples=100,num_chains=1)
    mcmc_inc.run(rng_key_, states_train, empirical_train_flat)

    # print the summary of the posterior distribution
    mcmc_inc.print_summary()

    # Get the MCMC samples and convert to a DataFrame
    posterior_inc = mcmc_inc.get_samples()
    df_inc = pd.DataFrame(posterior_inc)

    # Save the DataFrame to a CSV file
    df_inc.to_csv('../posterior_samples/production_posterior_test_5.csv', index=False)


def test():
    """
    Main function to run the script.
    """
    # Import dataset
    states_train, empirical_flat, empirical_seq, seq_mask, df, _, empirical_seq_flat  = import_dataset()
    uttSeq_list = jnp.unique(empirical_seq, axis=0)  # shape (U, L), U ≤ N

    # Get example state and utterance
    example_index = 2
    example_state = states_train[0:2]
    example_empirical = empirical_seq_flat[example_index]
    example_empirical_seq = empirical_seq[example_index]

    # Print example state and utterance
    print("Example state:", example_state)
    print("Example empirical utterance:", example_empirical)
    print("Example empirical utterance sequence:", example_empirical_seq)
    print("Seq to Flat mapping:", uttSeq_list)

    # Compute the incremental semantics for the example state
    example_incremental_semantics = incremental_semantics_jax(example_state, 0.95, 0.95, 0.5, 0.5)
    print("Example incremental semantics:", example_incremental_semantics)
    # Compute the global speaker for the example state
    example_global_speaker = global_speaker(example_state, 1.0, 0.95, 0.5)
    print("Example global speaker:", example_global_speaker)
    # Compute the incremental speaker for the example state
    example_incremental_speaker = incremental_speaker(example_state, 1.0, 0.95, 0.5)
    print("Example incremental speaker:", example_incremental_speaker)

    # Print the out of the likelihood function
    print("Likelihood function output:", likelihood_function_global_speaker(states_train, empirical_flat))
    
if __name__ == "__main__":
    #run_inference()
    test()