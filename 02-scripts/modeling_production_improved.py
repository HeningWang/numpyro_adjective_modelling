import argparse
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
import matplotlib.pyplot as plt
import numpyro
from numpyro.diagnostics import hpdi
import numpyro.distributions as dist
import numpyro.distributions.constraints as constraints
from numpyro import param, sample
from numpyro.distributions import constraints, HalfNormal, Normal, Uniform

from numpyro import handlers
from numpyro.infer import MCMC, NUTS, HMC, MixedHMC
from numpyro.infer import Predictive
from sklearn.model_selection import train_test_split
numpyro.set_platform("cpu")
numpyro.set_host_device_count(4)
print(jax.__version__)
print(jax.devices())

# ========================
# Global Variables (placeholders)
# ========================
utterance_list = None
utterance_prior = None
flat2categories = { "0": "D", 
                            "1": "DC", 
                            "2": "DCF",
                            "3": "DF",
                            "4": "DFC", 
                            "5": "C", 
                            "6": "CD", 
                            "7": "CDF",
                            "8": "CF",
                            "9": "CFD",
                            "10": "F",
                            "11": "FD",
                            "12": "FDC",
                            "13": "FC",
                            "14": "FCD"}

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
    file_path: str = "../01-dataset/01-production-data-preprocessed.csv",
    flag_output_empiricaldist_by_condition: bool = False
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
    df = df[df["conditions"].isin(["erdc", "zrdc", "brdc"])]

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

    if not flag_output_empiricaldist_by_condition:
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
    else:
        # If flag_output_empiricaldist_by_condition is True, return the empirical distribution by condition
        empirical_dist_by_condition = df.groupby(["item", "list",  "relevant_property", "sharpness"])["annotation_seq_flat"].value_counts(normalize=True).unstack(fill_value=0)
        empirical_dist_by_condition = jnp.array(empirical_dist_by_condition.values, dtype=jnp.float32)
        grouped_states = df.groupby(["item", "list", "relevant_property", "sharpness"])["statesArray"].first().reset_index(drop=True)
        states_array_by_condition = jnp.array(np.stack(grouped_states.values), dtype=jnp.float32)
        print("Empirical distribution by condition shape:", empirical_dist_by_condition.shape)
        #print("Empirical distribution by condition:", empirical_dist_by_condition)
        # Modify to return new grouped dataframe

        # Select only item, list, relevevance_property, sharpness and than group by item and list
        df_grouped = df[["relevant_property", "sharpness", "item", "list"]].copy()
        df_grouped = df_grouped.sort_values(by=["item", "list"])
        # Reduce df_grouped to unique rows
        df_grouped = (
            df_grouped.groupby(["item", "list", "relevant_property", "sharpness"], sort=False)
            .apply(lambda x: x.iloc[0])
            .reset_index(drop=True)
        )
        df_grouped["statesArray"] = states_array_by_condition.tolist()  # shape (n_conditions, 6, 3)
        # empirical_dist_by_condition is (54, 15)
        # Store empirical_dist_by_condition in long format, with one col for empirical_annotation and one for probability, add to df_grouped
        # First give it semantic names for each column

        print("Grouped DataFrame:")
        print(df_grouped)
        
        empirical_dist_df = pd.DataFrame(empirical_dist_by_condition, columns=[flat2categories[str(i)] for i in range(empirical_dist_by_condition.shape[1])])
        df_full = pd.concat([df_grouped.reset_index(drop=True), empirical_dist_df], axis=1)
       
        # 3. Convert to long format: melt category columns into two columns: 'annotation', 'probability'
        df_long = df_full.melt(
            id_vars=["relevant_property", "sharpness","item", "list", "statesArray"],
            value_vars=list(flat2categories.values()),
            var_name="empirical_annotation_category",
            value_name="probability"
        )
        print("Long format DataFrame:")
        print(df_long.head())
        return {
            "states_array_by_condition": states_array_by_condition,  # shape (n_conditions, 6, 3)
            "empirical_dist_by_condition": empirical_dist_by_condition,  # shape (n_conditions, n_utterances)
            "df": df_grouped,
            "df_long": df_long,  # long format DataFrame
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
        base_util = lax.cond(length == 1, lambda _: 2.0,
                     lambda _: lax.cond(length == 2, lambda _: 1.0, lambda _: 3.0, None), None)

        # Penalty for specific sequences
        penalty = lax.cond(is_penalized(u), lambda _: -3.0, lambda _: 0.0, None)

        # Boost if first token is 0 ("D")
        boost = lax.cond(u[0] == 0, lambda _: 2.0, lambda _: 1.0, None)

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
    return raw

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
                return prior * meaning_jax(token, states, prior, color_sem, form_sem, k, wf) # shape (n_obj,)
            
            posterior = lax.cond(token < 0, skip, apply, operand=None)
            return posterior, None

        final_belief, _ = lax.scan(step, state_prior, tokens[::-1])
        return final_belief

    # Apply over all utterances using vmap
    M = jax.vmap(apply_tokens)(utterances)  # shape (n_utt, n_obj)
    # Normalize the result
    M = normalize(M, axis=1)  # shape (n_utt, n_obj)
    return M

def global_speaker(
    states: jnp.ndarray,               # shape (n_objs,3)
    alpha: float = 1.0,
    color_semval: float = 0.95,
    k: float = 0.5,
    bias_subjectivity: float = 1,
    bias_length: float = 1,
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
        utt_prior = build_utterance_prior_jax(utterance_list=utterance_list, costParam_subjectivity=bias_subjectivity, costParam_length=bias_length) # from the global variable
    current_utt_prior = jnp.log(utt_prior)
    meaning_matrix = incremental_semantics_jax(states, color_semval, k)
    util_speaker = jnp.log(jnp.transpose(meaning_matrix)) + current_utt_prior
    softmax_result = jax.nn.softmax(alpha * util_speaker)
    return softmax_result

vectorized_global_speaker = jax.vmap(global_speaker, in_axes=(0, # states, along the first axis, i.e. one trial of the experiment
                                                              None, # alpha,
                                                              None, # color_semval
                                                              None, # k
                                                              None, # bias_subjectivity
                                                                None  # bias_length
                                                              )) 

def incremental_speaker(
    states: jnp.ndarray,
    alpha: float = 1.0,
    color_semval: float = 0.95,
    k: float = 0.5,
    bias_subjectivity: float = 1.0,
    bias_length: float = 1.0,
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
        utt_prior = build_utterance_prior_jax(utterance_list=utterance_list, costParam_subjectivity=bias_subjectivity, costParam_length=bias_length) # from the global variable
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
                posterior = global_speaker(states, alpha, color_semval, k, bias_subjectivity, bias_length, prior)
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

vectorized_incremental_speaker = jax.vmap(incremental_speaker, in_axes=(0, # states, along the first axis, i.e. one trial of the experiment
                                                              None, # alpha,
                                                              None, # color_semval
                                                              None, # k
                                                              None, # bias_subjectivity
                                                                None  # bias_length
                                                              )) 

def likelihood_function_global_speaker(states = None, empirical = None):
    # Initialize the parameter priors
    alpha = numpyro.sample("alpha", dist.HalfNormal(2))
    color_semval = numpyro.sample("color_semvalue", dist.Uniform(0, 1))
    k = numpyro.sample("k", dist.Uniform(0, 1))
    bias_subjectivity = numpyro.sample("bias_subjectivity", dist.Normal(0.0, 2.0))
    bias_length = numpyro.sample("bias_length", dist.Normal(0.0, 2.0))

    # Define the likelihood function
    with numpyro.plate("data", len(states)):
        # Get vectorized global speaker output for all states
        # For single output, it is shape (n_utt, n_obj)
        results_vectorized_global_speaker = vectorized_global_speaker(states, alpha, color_semval, k, bias_subjectivity, bias_length) #shape (nbatch_size, n_utt, n_obj)
        # Build a new probs vector of shape (n_states, n_utt), where the matix is reduced to the first row (referent index 0) along the second axis (n_obj)
        # We need the shape (nbatch_size, n_utt)
        utt_probs_conditionedReferent = results_vectorized_global_speaker[:,0,:] # Get the probs of utterances given the first state, referent is always the first state
        numpyro.deterministic("utt_probs_conditionedReferent", utt_probs_conditionedReferent) # Store the probs for later use
        dirichlet_concentration = utt_probs_conditionedReferent * 10
        probs = numpyro.sample("utterance_dist", dist.Dirichlet(dirichlet_concentration))
        if empirical is None:
            numpyro.sample("obs", dist.Categorical(probs=probs))
        else:
            numpyro.sample("obs", dist.Categorical(probs=probs), obs=empirical)

def likelihood_function_incremental_speaker(states = None, empirical = None):
    # Initialize the parameter priors
    alpha = numpyro.sample("alpha", dist.HalfNormal(2))
    color_semval = numpyro.sample("color_semvalue", dist.Uniform(0, 1))
    k = numpyro.sample("k", dist.Uniform(0, 1))
    bias_subjectivity = numpyro.sample("bias_subjectivity", dist.Normal(0.0, 2.0))
    #bias_length = numpyro.sample("bias_length", dist.Normal(0.0, 2.0))
    bias_length = -0.44

    # Define the likelihood function
    with numpyro.plate("data", len(states)):
        # Get vectorized incremental speaker output for all states
        # For single output, it is shape (n_utt, )
        results_vectorized_incremental_speaker = vectorized_incremental_speaker(states, alpha, color_semval, k, bias_subjectivity, bias_length) #shape (nbatch_size, n_utt)
        # Build a new probs vector of shape (n_states, n_utt), where the matix is reduced to the first row (referent index 0) along the second axis (n_obj)
        utt_probs_conditionedReferent = results_vectorized_incremental_speaker # shape (nbatch_size, n_utt)
        numpyro.deterministic("utt_probs_conditionedReferent", utt_probs_conditionedReferent) # Store the probs for later use
        dirichlet_concentration = utt_probs_conditionedReferent * 1
        probs = numpyro.sample("utterance_dist", dist.Dirichlet(dirichlet_concentration))
        if empirical is None:
            numpyro.sample("obs", dist.Categorical(probs=probs))
        else:
            numpyro.sample("obs", dist.Categorical(probs=probs), obs=empirical)

def svi_gb(states = None, empirical = None):
    """
    Define the model for SVI.
    This is a wrapper around the likelihood function.
    """
    # Use the global speaker likelihood function
    alpha = numpyro.sample("alpha", dist.HalfNormal(2.0))
    k = numpyro.sample("k", dist.Beta(1.0, 2.0))
    color_semval = numpyro.sample("color_semvalue", dist.Beta(2.0, 1.0))
    bias = numpyro.sample("bias_subjectivity", dist.Normal(0.0, 2.0))
    bias_length = numpyro.sample("bias_length", dist.Normal(0.0, 2.0))

    with numpyro.plate("data", len(states)):
        utt_probs = vectorized_global_speaker(states, alpha, color_semval, k, bias, bias_length)  # shape (n_states, n_utt)
        utt_probs = utt_probs[:, 0, :]  # assuming referent index 0
        eps = 1e-4
        numpyro.factor("loglik", jnp.sum(empirical * jnp.log(utt_probs + eps), axis=-1))

def svi_inc(states = None, empirical = None):
    """
    Define the model for SVI.
    This is a wrapper around the likelihood function.
    """
    # Use the global speaker likelihood function
    alpha = numpyro.sample("alpha", dist.HalfNormal(2.0))
    k = numpyro.sample("k", dist.Beta(1.0, 2.0))
    color_semval = numpyro.sample("color_semvalue", dist.Beta(2.0, 1.0))
    bias = numpyro.sample("bias", dist.Normal(0.0, 2.0))
    bias_length = -0.44 # Fixed value for bias_length, as per the original code

    with numpyro.plate("data", len(states)):
        utt_probs = vectorized_incremental_speaker(states, alpha, color_semval, k, bias, bias_length)  # shape (n_states, n_utt)
        eps = 1e-4
        numpyro.factor("loglik", jnp.sum(empirical * jnp.log(utt_probs + eps), axis=-1))



def svi_guide(states=None, empirical=None):
    # alpha ~ HalfNormal(2.0)
    alpha_scale = param("alpha_scale", 1, constraint=constraints.positive)
    sample("alpha", HalfNormal(alpha_scale))

    # k ~ Uniform(0.01, 1.0)
    k_loc = param("k_loc", 0.5, constraint=constraints.unit_interval)
    k_scale = param("k_scale", 0.1, constraint=constraints.positive)
    sample("k", Normal(k_loc, k_scale))  # No constraint because model clips it

    # color_semval ~ Uniform(0.01, 1.0)
    color_semval_loc = param("color_semval_loc", 0.9, constraint=constraints.unit_interval)
    color_semval_scale = param("color_semval_scale", 0.1, constraint=constraints.positive)
    sample("color_semvalue", Normal(color_semval_loc, color_semval_scale))

    # bias ~ Normal(0.0, 2.0)
    bias_loc = param("bias_loc", 0.0)
    bias_scale = param("bias_scale", 0.5, constraint=constraints.positive)
    sample("bias", Normal(bias_loc, bias_scale))

def run_svi(model = "gb"):
    """
    Run Stochastic Variational Inference (SVI) to estimate the parameters of the model.
    This function uses the svi_model and guide defined above.
    """
    rng_key = random.PRNGKey(42)  # Random key for reproducibility
    # Import dataset
    data = import_dataset(flag_output_empiricaldist_by_condition=True)
    states_train = data["states_array_by_condition"]  # shape (n_conditions, 6, 3)
    print("States train shape:", states_train.shape)
    empirical_train = data["empirical_dist_by_condition"]  # shape (n_conditions, n_utterances)
    df = data["df"]  # DataFrame with all the data
    df_long = data["df_long"]  # DataFrame in long format
    guide = numpyro.infer.autoguide.AutoNormal(svi_gb)
    print("States train shape:", states_train.shape)
    # Define the SVI kernel

    if model == "gb":
        model = svi_gb
    elif model == "inc":
        model = svi_inc

    # # =========================
    # # SVI kernel
    # svi_kernel = numpyro.infer.SVI(
    #     model= model,
    #     guide= guide,
    #     optim=numpyro.optim.Adam(step_size=1e-6),
    #     loss=numpyro.infer.Trace_ELBO()
    # )

    # # Run SVI
    # svi_result = svi_kernel.run(rng_key, num_steps=1000, states=states_train, empirical=empirical_train)

    # # Plot the losses
    # losses = svi_result.losses
    # plt.plot(losses)
    # plt.xlabel("Iteration")
    # plt.ylabel("ELBO Loss")
    # plt.title("SVI Loss over Training")
    # plt.grid(True)
    # plt.show()
    # # Print the summary of the posterior distribution
    # params = svi_result.params
    # print("SVI Result Parameters:")
    # print(params)
    # print("SVI Result Summary:")
    # for key, value in params.items():
    #     print(f"{key}: {value}")
    #     print(f"{key} shape: {value.shape}")

    # # Sample from guide (uses constrained values)
    # get_posterior_sample_predictive = Predictive(guide, params=svi_result.params, num_samples=len(states_train))
    # posterior_samples = get_posterior_sample_predictive(rng_key, states_train)
    # print("Posterior Samples:")
    # print(posterior_samples)
    # # Print means
    # for name, value in posterior_samples.items():
    #     print(f"{name}: mean = {value.mean():.3f}, std = {value.std():.3f}")

    # =========================
    # MCMC kernel
    kernel = numpyro.infer.NUTS(model, target_accept_prob=0.9, max_tree_depth=10)
    mcmc = MCMC(
        kernel,
        num_warmup=100,
        num_samples=250,
        num_chains=4,
        progress_bar=True
    )
    # Run MCMC
    mcmc.run(rng_key, states_train, empirical_train)
    mcmc.print_summary()
    # Get the MCMC samples
    posterior_samples = mcmc.get_samples()

    # Get posterior predictive samples
    if model == svi_gb:
        likelihood_function = likelihood_function_global_speaker
    elif model == svi_inc:
        likelihood_function = likelihood_function_incremental_speaker

    get_posterior_predictive = Predictive(likelihood_function, posterior_samples=posterior_samples)
    posterior_predictive = get_posterior_predictive(rng_key, states_train)

    # Export the posterior predictive results to a DataFrame
    utterance_probs = posterior_predictive.pop("utt_probs_conditionedReferent", None) # Shape (n_numsamples, n_states, n_utterances)
    mean_utterance_probs = utterance_probs.mean(axis=0)  # Shape (n_states, n_utterances)
    eps = 1e-4  # To avoid division by zero
    mean_utterance_probs = mean_utterance_probs + eps
    sd_utterance_probs = utterance_probs.std(axis=0)  # Shape (n_states, n_utterances)
    sd_utterance_probs = sd_utterance_probs + eps  # Avoid division by zero
    print("utterance_probs shape:", utterance_probs.shape)
    print("mean_utterance_probs shape:", mean_utterance_probs.shape)
    print("sd_utterance_probs shape:", sd_utterance_probs.shape)

    hard_labels_predictions = posterior_predictive.pop("obs", None)  # shape (n_numsamples, n_states)
    # Convert the utt_prob to long format and combine them with the DF
    # Using flat2categories to map the hard labels to categories
    
    df_utt_probs_mean = pd.DataFrame(
    mean_utterance_probs,
    columns=[flat2categories[str(i)] for i in range(mean_utterance_probs.shape[1])]
    )

    df_utt_probs_sd = pd.DataFrame(
        sd_utterance_probs,
        columns=[flat2categories[str(i)] for i in range(sd_utterance_probs.shape[1])]
    )
    # Copy the relevant columns from df_long
    df_long_copy = df_long[["item", "list","relevant_property", "sharpness"]].copy()

    print("df_long_copy", df_long_copy)

    # Convert to long format
    # Melt mean
    df_mean_long = pd.concat(
        [df_long_copy.reset_index(drop=True), df_utt_probs_mean],
        axis=1
    )
    df_mean_long = df_mean_long.melt(
        id_vars=["item", "list", "relevant_property", "sharpness"],
        value_vars=list(flat2categories.values()),
        var_name="utterance_category",
        value_name="probability"
    )

    # # Melt sd
    # df_sd_long = pd.concat(
    #     [df_long_copy.reset_index(drop=True), df_utt_probs_sd],
    #     axis=1
    # )
    # df_sd_long = df_sd_long.melt(
    #     id_vars=["relevant_property", "sharpness"],
    #     value_vars=list(flat2categories.values()),
    #     var_name="utterance_category",
    #     value_name="sd_probability"
    # )

    # # Merge mean and sd based on keys: relevant_property, sharpness, utterance_category
    # df_utt_probs_combined = pd.merge(
    #     df_mean_long,
    #     df_sd_long,
    #     on=["relevant_property", "sharpness", "utterance_category"]
    # )
    
    df_utt_probs_combined = df_mean_long.copy()

    print("df_utt_probs_combined", df_utt_probs_combined)
    # Combine the hard labels with the original DataFrame df_long
    df_utt_probs_combined = df_utt_probs_combined.rename(
        columns={"utterance_category": "annotation_category"}
    )
    
    df_long = df_long.rename(
        columns={"empirical_annotation_category": "annotation_category"}
    )

    # Add source
    df_utt_probs_combined["source"] = "model"
    df_long["source"] = "empirical"

    print(df_long.index.is_unique)   # Should be True
    print(df_utt_probs_combined.index.is_unique)   # Should be True

    # Combine the two DataFrames
    df_compare = pd.concat(
    [df_long.reset_index(drop=True), df_utt_probs_combined.reset_index(drop=True)],
    ignore_index=True
    )

    print("df_compare final output", df_compare)
    # Save the DataFrame to a CSV file
    model_name = "gb" if model == svi_gb else "inc"
    df_compare.to_csv(f"../05-modelling-production-data/production_posterior_{model_name}_relevanceXsharpness_SVI_long_soft.csv", index=False)

    # Transpose to (n_rows, n_samples)
    predictions_t = hard_labels_predictions.T  # jnp.transpose(predictions)

    
    # Convert to Python lists of floats (optional: jnp → np)
    pred_list = np.array(predictions_t).tolist()  # now a list of lists
    df["predictions"] = pred_list
    df.to_csv("../05-modelling-production-data/production_posteriorPredictive_gb_relevanceXsharpness_SVI_wide_hard.csv", index=False)
    #print("Posterior predictive shape:", posterior_predictive.shape)

    # ========================
def run_inference(speaker_type: str = "global",
                    num_warmup: int = 100,
                    num_samples: int = 250,
                    num_chains: int = 4,
                    save_predictive: bool = False,
                    output_file_name: str = "../05-modelling-production-data/production_posterior_gb_relevanceXsharpness_1.csv"
                  ):
    # Import dataset
    data = import_dataset()
    states_train = data["states_train"]
    empirical_train_flat = data["empirical_flat"]
    empirical_train_seq_flat = data["empirical_seq_flat"]

    # Some printing for debugging
    print("States train shape:", states_train.shape)
    print("Empirical train flat shape:", empirical_train_flat.shape)
    print("Output file name:" , output_file_name)

    # Define the MCMC kernel and the number of samples
    rng_key = random.PRNGKey(4711)
    rng_key, rng_key_ = random.split(rng_key)

    if speaker_type == "global":
        # Use the global speaker likelihood function
        kernel = NUTS(likelihood_function_global_speaker, target_accept_prob=0.9, max_tree_depth=10)
    elif speaker_type == "incremental":
        # Use the incremental speaker likelihood function
        kernel = NUTS(likelihood_function_incremental_speaker, target_accept_prob=0.9, max_tree_depth=10)
    #kernel = MixedHMC(HMC(likelihood_function, trajectory_length=1.2), num_discrete_updates=20)
    mcmc_inc = MCMC(kernel, num_warmup=num_warmup,num_samples=num_samples, num_chains=num_chains)
    mcmc_inc.run(rng_key_, states_train, empirical_train_seq_flat)

    # print the summary of the posterior distribution
    mcmc_inc.print_summary()

    # Get the MCMC samples and convert to a DataFrame
    posterior_inc = mcmc_inc.get_samples() 

    # Save other, but not save the key "utt_probs_conditionedReferent" to the DataFrame
    # Instead, save it use jnp.savez
    if "utt_probs_conditionedReferent" in posterior_inc:
        utt_probs_conditionedReferent = posterior_inc.pop("utt_probs_conditionedReferent")
        utt_dist = posterior_inc.pop("utterance_dist", None)  # Remove utterance_dist if it exists
        probs_file_name = output_file_name.replace(".csv", "_utt_probs_conditionedReferent.npz")
        jnp.savez(probs_file_name, utt_probs_conditionedReferent=utt_probs_conditionedReferent)
    else:
        print("Warning: 'utt_probs_conditionedReferent' not found in posterior samples.")

    df_inc = pd.DataFrame(posterior_inc)

    # Save the DataFrame to a CSV file
    df_inc.to_csv(output_file_name, index=False)

    if save_predictive:
        # Create a predictive model
        if speaker_type == "global":
            # Use the global speaker likelihood function for predictions
            predictive = Predictive(likelihood_function_global_speaker, mcmc_inc.get_samples())
        elif speaker_type == "incremental":
            predictive = Predictive(likelihood_function_incremental_speaker, mcmc_inc.get_samples())
        # Generate predictions
        predictions = predictive(rng_key_, states_train)["obs"]
        # Convert predictions to a DataFrame
        df_predictions = pd.DataFrame(predictions)
        # Save the predictions to a CSV file
        df_predictions.to_csv(output_file_name.replace(".csv", "_predictions.csv"), index=False)


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
    example_index = 3
    example_state = states_train[2]
    example_empirical = empirical_train_seq_flat[example_index]
    example_empirical_seq = empirical_train_seq[example_index]

    # Print example state and utterance
    print("Example state:", example_state)
    # print("Example empirical utterance:", example_empirical)
    # print("Example empirical utterance sequence:", example_empirical_seq)
    # print("Seq to Flat mapping:", uttSeq_list)

    # Compute the incremental semantics for the example state
    example_incremental_semantics = incremental_semantics_jax(example_state, 0.95, 0.95, 0.5, 0.5)
    print("Example incremental semantics:", example_incremental_semantics)
    # Compute the global speaker for the example state
    example_global_speaker = global_speaker(example_state, 0.5, 0.95, 0.5)
    print("Example global speaker:", example_global_speaker[0,])
    # Compute the incremental speaker for the example state
    example_incremental_speaker = incremental_speaker(example_state, 0.5, 0.95, 0.5)
    print("Example incremental speaker:", example_incremental_speaker)
    #print("Example incremental speaker shape:", example_incremental_speaker.shape)

    # example_states_array = states_train[0:2]
    # # Test the vectorized global speaker
    # example_vectorized_global_speaker = vectorized_global_speaker(example_states_array, 1, 0.95, 0.5, 1)
    # utt_probs_conditionedReferent = example_vectorized_global_speaker[:,0,:] # Get the probs of utterances given the first state, referent is always the first state
    # print("Example vectorized global speaker distilled result shape:", utt_probs_conditionedReferent.shape)
    # print("Example vectorized global speaker distilled result shape:", utt_probs_conditionedReferent)

    # # Test the vectorized incremental speaker
    # example_vectorized_incremental_speaker = vectorized_incremental_speaker(example_states_array, 1, 0.95, 0.5, 1)
    # utt_probs_conditionedReferent = example_vectorized_incremental_speaker # Get the probs of utterances given the first state, referent is always the first state
    # print("Example vectorized incremental speaker distilled result shape:", utt_probs_conditionedReferent.shape)
    # print("Example vectorized incremental speaker distilled result shape:", utt_probs_conditionedReferent)

def test_import_dataset():
    """
    Test the import_dataset function.
    """
    data = import_dataset(flag_output_empiricaldist_by_condition=True)
    states_train = data["states_array_by_condition"]
    empirical_train = data["empirical_dist_by_condition"]

    example_states = states_train[:5]
    example_empirical = empirical_train[:5]
    print("States train shape:", states_train.shape)
    print("Empirical train flat shape:", empirical_train.shape)
    print("Empirical train seq flat shape:", empirical_train.shape)
    print(example_states)
    print(example_empirical)

    

    result_incremental_speaker = vectorized_incremental_speaker(example_states, 1.0, 0.95, 0.5, 1.0, 1.0)
    print("Result incremental speaker:", result_incremental_speaker)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run speaker inference with NumPyro.")
    parser.add_argument("--speaker_type", type=str, choices=["global", "incremental"], default="global",
                        help="Choose the speaker model type.")
    parser.add_argument("--output_file_name", type=str, default="../posterior_samples/posterior_samples.csv",
                        help="Path to save posterior samples.")
    parser.add_argument("--num_samples", type=int, default=20, help="Number of posterior samples.")
    parser.add_argument("--num_warmup", type=int, default=20, help="Number of warm-up iterations.")
    parser.add_argument("--num_chains", type=int, default=4, help="Number of MCMC chains.")
    parser.add_argument("--save_predictive", action="store_true",
                        help="Include this flag to save posterior predictive outputs.")
    parser.add_argument("--test", action="store_true", help="Run test function and exit.")

    args = parser.parse_args()

    if args.test:
        run_svi("inc")
        #test()
        #test_import_dataset()
    else:
        run_inference(
            speaker_type=args.speaker_type,
            output_file_name=args.output_file_name,
            num_samples=args.num_samples,
            num_warmup=args.num_warmup,
            num_chains=args.num_chains,
            save_predictive=args.save_predictive
        )