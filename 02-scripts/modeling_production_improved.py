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
print(jax.devices())

# ========================
# Global Variables (placeholders)
# ========================
utterance_list = None
utterance_prior = None

# ========================
# Helpers
# ========================

def normalize(arr, axis=1):
    """
    Normalize arr along axis.
    Args:
        arr: jnp.ndarray to normalize
        axis: axis to normalize along
    Returns:
        jnp.ndarray: normalized array
    """
    return arr / jnp.sum(arr, axis=axis, keepdims=True)

def import_dataset(
    file_path: str = "../01-dataset/01-production-data-preprocessed.csv"
    ):
    """
    Load the dataset and preprocess it.
    - Load the data from a CSV file
    - Drop missing values in the 'annotation' column
    - Subset the data by conditions to include only conditions involving relevant manipulation to size adjectives
    - Encode states via vectorized slicing
    - Encode utterances using different methods:
        1) flat codes for utterances
        2) build a sequence of symbols for each utterance
        3) build a mask array to ignore padding
        4) build a flat codes for the utterances given the sequence string
    - Add all encodings to the DataFrame
    - Return a dictionary with all the relevant data
    Args:
        file_path: path to the CSV file
    Returns:    
        A dictionary with the following keys:
            - states_train: jnp.ndarray of shape (N, 6, 3)
            - empirical_flat: jnp.ndarray of shape (N,)
            - empirical_seq: jnp.ndarray of shape (N, max_len)
            - seq_mask: jnp.ndarray of shape (N, max_len)
            - df: DataFrame with all the relevant data
            - unique_utterances: jnp.ndarray of shape (U, max_len)
            - empirical_seq_flat: jnp.ndarray of shape (N,)
    """
    # Load & drop missing
    df = pd.read_csv(file_path)
    df = df.dropna(subset=["annotation"]).copy()

    # Subset the data by conditions to include only conditions involving relevant manipulation to size adjectives
    df = df[df["conditions"].isin(["erdc", "erdf"])]

    # Encode states via vectorized slicing
    #    – sizes are numeric already
    #    – colors: "blue"→1, else 0
    #    – forms:  "circle"→1, else 0
    sizes = df.iloc[:, 6:12].to_numpy(dtype=float)                  # (N,6)
    colors = (df.iloc[:, 12:18] == "blue").to_numpy(dtype=int)      # (N,6)
    forms  = (df.iloc[:, 18:24] == "circle").to_numpy(dtype=int)    # (N,6)

    # Stack into shape (N,6,3)
    states_np = np.stack([sizes, colors, forms], axis=2) 
    states_train = jnp.array(states_np)                # (N,6,3)

    # Encode uttrances using different methods
    # 1) flat codes for utterances
    # Make it categorical from strings to integers, each unique string is a category
    df["annotation"] = df["annotation"].astype("category")
    cats = list(df["annotation"].cat.categories)         # e.g. ["D","C","CD",...]
    
    # Build maps
    utt2idx: Dict[str,int] = {u:i for i,u in enumerate(cats)}
    idx2utt: Dict[int,str] = {i:u for u,i in enumerate(cats)}
    
    # Stack into shape (N,)
    empirical_flat = jnp.array(df["annotation"].cat.codes.to_numpy(), dtype=jnp.int32) # (N,)
    

    # 2) Build a sequence of symbols for each utterance
    # D = 0, C = 1, F = 2, padding = -1
    # e.g. DCF is [0,1,2] and D is [0, -1, -1]
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

    # Stack into one jnp.ndarray
    empirical_seq = jnp.array(sequences, dtype=jnp.int32)  # shape (N, max_len)

    # Get the unique utterances string from empirical_seq
    unique_utterances = jnp.unique(empirical_seq, axis=0)  # shape (U, max_len)

    # 3) Build a mask array to ignore padding
    seq_mask = jnp.array(
        [[True]*len(u) + [False]*(max_len - len(u)) for u in utt_strings],
        dtype=bool
    )  # shape (N, max_len)

    # 4) Build a flat codes for the utterances given the sequence string
    # Build a dict mapping each unique utterance to its index
    uttSeq2idx = {tuple(u): i for i, u in enumerate(unique_utterances.tolist())}
    # Map each utterance in the sequence to its index
    empirical_seq_flat = jnp.array(
        [uttSeq2idx[tuple(u)] for u in empirical_seq.tolist()],
        dtype=jnp.int32
    )  # shape (N,)

    # Add all encodings to the DataFrame
    df["statesArray"] = states_train.tolist()  # shape (N,)
    df["annotation_string_flat"] = empirical_flat.tolist()  # shape (N,)
    df["annotation_seq"] = empirical_seq.tolist()  # shape (N,)
    df["annotation_seq_mask"] = seq_mask.tolist()  # shape (N, max_len)
    df["annotation_seq_flat"] = empirical_seq_flat.tolist()  # shape (N,)

    # Return a dictionary with all the relevant data
    return {
        "states_train": states_train,
        "empirical_flat": empirical_flat,
        "empirical_seq": empirical_seq,
        "seq_mask": seq_mask,
        "df": df,
        "unique_utterances": unique_utterances,
        "empirical_seq_flat": empirical_seq_flat,
    }

def build_utterance_prior(utterance_list: List[str],
                        costParam_length: float = 1.0,
                        costParam_bias: float = 1.0,
                        costParam_subjectivity: float = 1.0
                          ) -> jnp.ndarray:
    """
    Build a prior over utterances based on their length and specific biases.
    The utility values are computed as a linear combination of:
    - Length-based utility
    - Bias-based utility
    - Subjectivity-based utility
    The final utility is then transformed into a probability distribution using softmax.
    The higher the utility, the higher the probability of selecting that utterance. 
    A penalization is applied with negative values, and a boost is applied with positive values.
    Args:
        utterance_list: list of unique utterances
        costParam_length: weight for length-based utility
        costParam_bias: weight for bias-based utility
        costParam_subjectivity: weight for subjectivity-based utility
    Returns:
        jnp.ndarray: prior over utterances
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

def build_utterance_prior_jax(
    utterance_list: jnp.ndarray,  # shape (U, 3)
    costParam_length: float = 1.0,
    costParam_bias: float = 1.0,
    costParam_subjectivity: float = 1.0
) -> jnp.ndarray:
    """
    Build a prior over utterances using JAX-native operations.
    Assumes utterance_list is a (U, 3) jnp.ndarray where -1 is padding.
    """

    penalized = jnp.array([
        [1, 0, -1],  # "CD"
        [2, 1, -1],  # "FD"
        [1, 2, 0],   # "CFD"
        [2, 1, 0],   # "FCD"
        [2, 0, 1]    # "FDC"
    ])  # shape (5, 3)

    def is_penalized(u):
        return jnp.any(jnp.all(u == penalized, axis=1))

    def compute_utils(carry, u):
        # Length utility: count of non-padding elements
        length = jnp.sum(u >= 0)
        base_util = lax.cond(length == 1, lambda _: 3.0,
                     lambda _: lax.cond(length == 2, lambda _: 2.0, lambda _: 1.0, None), None)

        # Penalty for specific sequences
        penalty = lax.cond(is_penalized(u), lambda _: -0.5, lambda _: 0.0, None)

        # Boost if first token is 0 ("D")
        boost = lax.cond(u[0] == 0, lambda _: 1.0, lambda _: 0.0, None)

        # Combine with weights
        total_util = (
            costParam_length * base_util +
            costParam_bias * penalty +
            costParam_subjectivity * boost
        )

        return carry, total_util

    # Scan over all utterances
    _, utils = lax.scan(compute_utils, None, utterance_list)

    return jax.nn.softmax(utils)

# ========================
# Global Variables (Setup)
# ========================
utterance_list = import_dataset()["unique_utterances"]  # shape (U,3)
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
    # Use half the objects as the sample size
    sample_size = int(round(states.shape[0] / 2)) # Sample size is half of the number of objects in a given context
    # Create a categorical distribution with the given probabilities
    costum_dist = dist.Categorical(probs=states_prior)
    # Sample indices from the distribution
    sample_indices = jnp.unique(costum_dist.sample(random.PRNGKey(0),(1,sample_size)), size= sample_size)
    # Get the sampled states
    sorted_states = states[sample_indices][:,0]

    # Compute the threshold with k-procent semantics
    min_val = jnp.min(sorted_states)
    max_val = jnp.max(sorted_states)
    weighted_threshold = max_val - k * (max_val - min_val)

    return weighted_threshold

def meaning_jax(
    word: int,
    states: jnp.ndarray,
    state_prior: jnp.ndarray,
    color_semval: float,
    form_semval: float,
    k: float,
    wf: float,
) -> jnp.ndarray:
    """
    Compute the meaning of a word given the states and their prior.
    Args:
        word: the word to compute the meaning for.
        states: the states of the objects.
        state_prior: the prior distribution over states.
        color_semval: semantic value for color.
        form_semval: semantic value for form.
        k: weight factor for size semantics.
        wf: weight factor for size semantics.
    Returns:
        A jnp.ndarray of shape (n_obj,) with the meaning of the word.
    """
    
    sizes = states[:, 0]
    colors = states[:, 1]
    forms  = states[:, 2]

    # Compute the semantic value for size
    def size_case(_):
        threshold = get_threshold_kp_sample_jax(states, state_prior, k)
        return jax.vmap(size_semantic_value, in_axes=(0, None, None))(states, threshold, wf)[:, 0]

    # Compute the semantic value for color and form
    def color_case(_):
        return jnp.where(colors == 1, color_semval, 1.0 - color_semval)

    def form_case(_):
        return jnp.where(forms == 1, form_semval, 1.0 - form_semval)
 
    # Compute the semantic value for padding
    def padding_case(_):
        return jnp.ones(states.shape[0], dtype=jnp.float32)

    # remap -1 (padding) to 3
    index = jnp.where(word == -1, 3, word)

    # Compute the semantic value for each case
    raw = lax.switch(
        index,
        [size_case, color_case, form_case, padding_case],
        operand=None
    ) # shape (n_obj,)

    # Normalize the result
    return normalize(raw, axis=0)

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
    At each step, we are dealing with marginal distribution over states given one utterance.
    Stack the results for all utterances together, we have a joint distribution over states and utterances.
    Args:
        states: the states of the objects.
        color_sem: semantic value for color.
        form_sem: semantic value for form.
        k: weight factor for size semantics.
        wf: weight factor for size semantics.
        state_prior: prior distribution over states.
    Returns:
        A jnp.ndarray of shape (n_utt, n_obj) with the meaning of the utterances.
    """
    utterances = utterance_list  # assumed global
    n_obj = states.shape[0]
    if state_prior is None:
        state_prior = jnp.ones(n_obj) / n_obj  # uniform prior, shape (n_obj,)

    def apply_tokens(tokens: jnp.ndarray) -> jnp.ndarray:
        def step(prior, token):
            def skip(_):
                return prior # shape (n_obj,)
            def apply(_):
                return meaning_jax(token, states, prior, color_sem, form_sem, k, wf) # shape (n_obj,)
            
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
    bias: float = 1,
    utt_prior: jnp.ndarray = None
):
    """
    Output: P(utterance | referent) using global RSA semantics.
    For each states, compute the meaning of the utterances using the incremental semantics.
    Then, compute the utility of the speaker using the meaning of the utterances and the prior.

    Args:
        states: the states of the objects.
        alpha: weight for the speaker utility.
        color_semval: semantic value for color.
        k: weight factor for size semantics.
        utt_prior: prior distribution over utterances.
        bias: cost parameter for subjectivity.
    Returns:
        A jnp.ndarray of shape (n_obj, n_utt) with the meaning of the utterances.
    """
    if utt_prior is None:
        utt_prior = build_utterance_prior_jax(utterance_list=utterance_list, costParam_subjectivity=bias) # from the global variable
    current_utt_prior = jnp.log(utt_prior)
    meaning_matrix = incremental_semantics_jax(states, color_semval, k)
    util_speaker = jnp.log(jnp.transpose(meaning_matrix)) + current_utt_prior
    softmax_result = jax.nn.softmax(alpha * util_speaker)
    return softmax_result

vectorized_global_speaker = jax.vmap(global_speaker, in_axes=(0, # states, along the first axis, i.e. one trial of the experiment
                                                              None, # alpha,
                                                              None, # color_semval
                                                              None, # k
                                                              None # bias
                                                              )) 

def incremental_speaker(
    states: jnp.ndarray,
    alpha: float = 1.0,
    color_semval: float = 0.95,
    k: float = 0.5,
    bias: float = 1,
    utt_prior: jnp.ndarray = None
):
    """
    Output: P(utterance | referent) using incremental RSA semantics:
    for each utterance, process one token at a time, updating the utterance
    prior using global_speaker after each step.
    
    Returns:
        Matrix of shape (n_utt, n_obj) with utterance probabilities.
    """
    utterances = utterance_list  # assumed global
    n_objs = states.shape[0]
    if utt_prior is None:
        # Reshape for broadcasting
        utt_prior = build_utterance_prior_jax(utterance_list=utterance_list, costParam_subjectivity=bias)
        utt_prior = utt_prior.reshape(1, -1)  # shape (1, n_utt)
        utt_prior = jnp.tile(utt_prior, (n_objs, 1))  # shape (n_obj, n_utt)

    def apply_tokens(tokens: jnp.ndarray) -> jnp.ndarray:
        """
        Given a single utterance (shape (3,)), compute posterior distribution over objects.
        Applies global_speaker(...) incrementally from left to right using lax.scan.
        """

        def step(current_speaker_matrix, token):
            # Skip padding (-1): return unchanged prior
            def skip(_):
                return current_speaker_matrix # shape (n_obj, n_utt)

            def apply(_):
                # Get the posterior distribution over utterances given the current token
                prior = current_speaker_matrix[0,:]  
                posterior = global_speaker(states, alpha, color_semval, k, bias, prior)
                return posterior # shape (n_obj, n_utt)


            new_speaker_matrix = lax.cond(token < 0, skip, apply, operand=None) # shape (n_obj, n_utt)
            return new_speaker_matrix, None  # carry prior forward # shape (n_obj, n_utt)

        final_speaker_matrix, _ = lax.scan(step, utt_prior, tokens)
        
        return final_speaker_matrix

    # Apply over all utterances using vmap
    M = jax.vmap(apply_tokens)(utterances)  # shape (n_utt, n_obj, n_utt)

    # Should be a speaker matrix shape (n_obj, n_utt)
    # Along the first axis, we extract the n_utt-th slice of the first row
    # For example, for the first utterance, we take the (1,1,1) slice
    # For the second utterance, we take the (2,1,2) slice
    # For the third utterance, we take the (3,1,3) slice
    # 2. Vectorized extraction: get M[i, 0, i] for all i
    idx = jnp.arange(M.shape[0])            # [0, 1, 2, ..., n_utt - 1]
    final_probs  = M[idx, 0, idx]          # shape: (n_utt,)
    #final_probs = normalize(final_probs, axis=0)  # normalize to sum=1
    meaning_matrix = incremental_semantics_jax(states, color_semval, k)
    util_speaker = jnp.log(jnp.transpose(meaning_matrix)) + final_probs
    softmax_result = jax.nn.softmax(alpha * util_speaker)
    return final_probs

vectorized_incremental_speaker = jax.vmap(incremental_speaker, in_axes=(0, # states,
                                                                        None, # alpha
                                                                        None, # color_semval
                                                                        None, #k
                                                                        None  # bias
                                                                        ))


def likelihood_function_global_speaker(states = None, empirical = None):
    # Initialize the parameter priors
    alpha = numpyro.sample("alpha", dist.HalfNormal(5))
    color_semval = numpyro.sample("color_semvalue", dist.Uniform(0, 1))
    k = numpyro.sample("k", dist.Uniform(0.01, 1))
    bias = numpyro.sample("bias", dist.HalfNormal(5))

    # Define the likelihood function
    with numpyro.plate("data", len(states)):
        # Get vectorized global speaker output for all states
        # For single output, it is shape (n_utt, n_obj)
        results_vectorized_global_speaker = vectorized_global_speaker(states, alpha, color_semval, k, bias) #shape (nbatch_size, n_utt, n_obj)
        # Build a new probs vector of shape (n_states, n_utt), where the matix is reduced to the first row (referent index 0) along the second axis (n_obj)
        # We need the shape (nbatch_size, n_utt)
        utt_probs_conditionedReferent = results_vectorized_global_speaker[:,0,:] # Get the probs of utterances given the first state, referent is always the first state
        if empirical is None:
            numpyro.sample("obs", dist.Categorical(probs=utt_probs_conditionedReferent))
        else:
            numpyro.sample("obs", dist.Categorical(probs=utt_probs_conditionedReferent), obs=empirical)

def likelihood_function_incremental_speaker(states = None, empirical = None):
    # Initialize the parameter priors
    mu = jnp.array([1.0, 1.0])  # prior means for alpha and bias
    cov = jnp.array([[0.25, -0.15], [-0.15, 0.25]])  # prior cov matrix, tuned to your correlation
    z = numpyro.sample("alpha_bias", dist.MultivariateNormal(mu, covariance_matrix=cov))
    alpha, bias = z[0], z[1]
    color_semval = numpyro.sample("color_semvalue", dist.Uniform(0, 1))
    k = numpyro.sample("k", dist.Uniform(0, 1))
    # Define the likelihood function
    with numpyro.plate("data", len(states)):
        # Get vectorized incremental speaker output for all states
        # For single output, it is shape (n_utt, )
        results_vectorized_incremental_speaker = vectorized_incremental_speaker(states, alpha, color_semval, k, bias) #shape (nbatch_size, n_utt)
        # Build a new probs vector of shape (n_states, n_utt), where the matix is reduced to the first row (referent index 0) along the second axis (n_obj)
        utt_probs_conditionedReferent = results_vectorized_incremental_speaker # shape (nbatch_size, n_utt)
        if empirical is None:
            numpyro.sample("obs", dist.Categorical(probs=utt_probs_conditionedReferent))
        else:
            numpyro.sample("obs", dist.Categorical(probs=utt_probs_conditionedReferent), obs=empirical)

def run_inference():
    #states_train, empirical_train_flat, empirical_train_seq, _, _, _, empirical_train_seq_flat = import_dataset()
    data = import_dataset()
    states_train = data["states_train"]
    empirical_train_seq = data["empirical_seq"]
    empirical_train_flat = data["empirical_flat"]
    empirical_train_seq_flat = data["empirical_seq_flat"]
    print("States train shape:", states_train.shape)
    print("Empirical train flat shape:", empirical_train_flat.shape)
    output_file_name = "../posterior_samples/production_posterior_full_inc_10k_4p_alphabiasdependent.csv"
    print("Output file name:" , output_file_name)
    # define the MCMC kernel and the number of samples
    rng_key = random.PRNGKey(11)
    rng_key, rng_key_ = random.split(rng_key)

    kernel = NUTS(likelihood_function_incremental_speaker)
    #kernel = MixedHMC(HMC(likelihood_function, trajectory_length=1.2), num_discrete_updates=20)
    mcmc_inc = MCMC(kernel, num_warmup=1000,num_samples=2500, acceptance_rate_target=0.9, num_chains=4)
    mcmc_inc.run(rng_key_, states_train, empirical_train_seq_flat)

    # print the summary of the posterior distribution
    mcmc_inc.print_summary()

    # Get the MCMC samples and convert to a DataFrame
    posterior_inc = mcmc_inc.get_samples()
    df_inc = pd.DataFrame(posterior_inc)

    # Save the DataFrame to a CSV file
    df_inc.to_csv(output_file_name, index=False)


def test():
    """
    Main function to run the script.
    """
    # Import dataset
    data = import_dataset()
    states_train = data["states_train"]
    empirical_train_seq = data["empirical_seq"]
    empirical_train_flat = data["empirical_flat"]
    empirical_train_seq_flat = data["empirical_seq_flat"]
    uttSeq_list = jnp.unique(empirical_train_seq, axis=0)  # shape (U, L), U ≤ N

    # Get example state and utterance
    example_index = 2
    example_state = states_train[2]
    example_empirical = empirical_train_seq_flat[example_index]
    example_empirical_seq = empirical_train_seq[example_index]

    # Print example state and utterance
    print("Example state:", example_state)
    print("Example empirical utterance:", example_empirical)
    print("Example empirical utterance sequence:", example_empirical_seq)
    print("Seq to Flat mapping:", uttSeq_list)

    # Compute the incremental semantics for the example state
    example_incremental_semantics = incremental_semantics_jax(example_state, 0.95, 0.95, 0.5, 0.5)
    print("Example incremental semantics:", example_incremental_semantics)
    # Compute the global speaker for the example state
    example_global_speaker = global_speaker(example_state, 0.5, 0.95, 0.5)
    print("Example global speaker:", example_global_speaker)
    # Compute the incremental speaker for the example state
    example_incremental_speaker = incremental_speaker(example_state, 0.5, 0.95, 0.5)
    print("Example incremental speaker:", example_incremental_speaker)
    print("Example incremental speaker shape:", example_incremental_speaker.shape)

    example_states_array = states_train[0:2]
    # Test the vectorized global speaker
    example_vectorized_global_speaker = vectorized_global_speaker(example_states_array, 1, 0.95, 0.5, 1)
    utt_probs_conditionedReferent = example_vectorized_global_speaker[:,0,:] # Get the probs of utterances given the first state, referent is always the first state
    print("Example vectorized global speaker distilled result shape:", utt_probs_conditionedReferent.shape)
    print("Example vectorized global speaker distilled result shape:", utt_probs_conditionedReferent)

    # Test the vectorized incremental speaker
    example_vectorized_incremental_speaker = vectorized_incremental_speaker(example_states_array, 1, 0.95, 0.5, 1)
    utt_probs_conditionedReferent = example_vectorized_incremental_speaker # Get the probs of utterances given the first state, referent is always the first state
    print("Example vectorized incremental speaker distilled result shape:", utt_probs_conditionedReferent.shape)
    print("Example vectorized incremental speaker distilled result shape:", utt_probs_conditionedReferent)
if __name__ == "__main__":
    run_inference()
    #test()