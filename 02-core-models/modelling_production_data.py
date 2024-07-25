import os

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

import numpyro
from numpyro.diagnostics import hpdi
import numpyro.distributions as dist
from numpyro import handlers
from numpyro.infer import MCMC, NUTS
from sklearn.model_selection import train_test_split
numpyro.set_platform("cpu")

print(jax.__version__)
jax.devices()

def import_dataset(file_path = "../01-dataset/01-production-data-preprocessed.csv"):

    def encode_states(line):
      states = []
      for i in range(6):
        color = 1 if line.iloc[12 + i] == "blue" else 0
        form = 1 if line.iloc[18 + i] == "circle" else 0
        new_obj = (line.iloc[6 + i], color, form) # size, color, form
        states.append(new_obj)
      return jnp.array(states)

    def encode_empirical(utterances_list):
        """
        Input: list of strings of 15 categories
        Output: jnp.array of indices
        Given the ordering of utterances, encode the strings into indices.
        """
        # Define the fixed ordering of utterances
        utterances_order = [
            "D", "C", "F", "CD", "CF", "DC", "DF",
            "FC", "FD", "DCF", "DFC", "CDF", "CFD", "FCD", "FDC"
        ]
        
        # Create a dictionary that maps each utterance to its index
        utterance_to_index = {utterance: idx for idx, utterance in enumerate(utterances_order)}
        
        # Encode the input list of strings into indices
        indices = [utterance_to_index[utterance] for utterance in utterances_list]
        
        # Return the indices as a jnp.array
        return jnp.array(indices)

    # Import the data
    df = pd.read_csv(file_path)

    # Mutate the dataset to include the states of the objects
    df["states"] = df.apply(lambda row: encode_states(row), axis=1)

    # split the dataset into training and test sets
    #train, test = train_test_split(df, test_size=0.99, random_state=42)
    # Use the whole dataset as training set
    train = df
    train = train.dropna(subset=['annotation'])
    
    states_train = jnp.stack([cell for cell in train.states])
    empirical_train = encode_empirical(train.annotation.tolist())

    train["annotation_encoded"] = empirical_train

    return states_train, empirical_train, train

# Define lexicons
colors = ["red", "blue"]
sizes = ["big", "small"]
forms = ["square", "circle"]

# define dictionary of utterances and their costs
def classify_utterance(utterance):
    if len(utterance.split()) <= 1:
        categorie = "D" if utterance in sizes else "C" if utterance in colors else "F"
        return categorie
    else:
        words = utterance.split()
        categorie = "".join(["D" if x in sizes else "C" if x in colors else "F" for x in words])
        return categorie



def cost(utterance, color_cost=1, size_cost=1, form_cost=1, costWeight=0.5): # TODO: Add those parameter to the model inference?
    if len(utterance.split()) <= 1:
        cost = size_cost if utterance in sizes else color_cost if utterance in colors else form_cost
        return cost * costWeight
    else:
        words = utterance.split()
        cost = sum([size_cost if x in sizes else color_cost if x in colors else form_cost for x in words])
        return cost * costWeight

utterances = [
            "D",
            "C",
            "F",
            "CD", #CD
            "CF",
            "DC",
            "DF",
            "FC",
            "FD", #FD
            "DCF",
            "DFC",
            "CDF",
            "CFD", # CFD
            "FCD", # FCD
            "FDC", # FDC
]




def utterance_prior(utterances):
    """
    Input: list of utils scores
    Output: list of prior probabilities for each utterance
    """
    def utterance_utils(biased=False):
        """
        Input: list of utterances
        Output: list of utils scores for each utterance

        Depends on the length of utterances, assign 3 to the ones with length 1, 2 to the ones with length 2 and 1 to the ones with length 3
        Also, allow to costumize the utils scores given the value of the utterances
        If biased = True, assign 0.5 to "CD, FD, CFD, FCD, FDC", assign +1 to utterances staring with "D" on top of the previous rule
        """
        # Calculate the utils scores based on the length of utterances
        utils = [3 if len(utt) == 1 else 2 if len(utt) == 2 else 1 for utt in utterances]

        # Customize the utils scores if biased is True
        if biased:
            biased_utils = [0.5 if utterances[i] == "CD" or 
                            utterances[i] == "FD" or 
                            utterances[i] == "CFD" 
                            or utterances[i] == "FCD" 
                            or utterances[i] == "FDC" else utils[i] for i in range(len(utterances))]
            utils = [1 + biased_utils[i] if utterances[i].startswith("D") else biased_utils[i] for i in range(len(utterances))]

        utils = jnp.array(utils)
        return utils

    utils = utterance_utils(biased = True)
    prior = jnp.exp(utils) / jnp.sum(jnp.exp(utils))
    return prior

def normalize(arr, axis=1):
    """
    Normalize arr along axis
    """
    return arr / jnp.sum(arr, axis=axis, keepdims=True)

def uniform_state_prior(nobj=6):
    """
    Input: number of objects
    Output: list of prior probabilities for each object
    """
    prior=normalize(jnp.ones((2,nobj)))
    return prior


def get_threshold_kp_sample_jax(states, states_prior, k=0.5):
    sample_size = int(round(states.shape[0] / 2)) # Sample size is half of the number of objects in a given context
    costum_dist = dist.Categorical(probs=states_prior)
    sample_indices = jnp.unique(costum_dist.sample(random.PRNGKey(0),(1,sample_size)), size= sample_size)
    sorted_states = states[sample_indices][:,0]
    min_val = jnp.min(sorted_states)
    max_val = jnp.max(sorted_states)

    weighted_threshold = max_val - k * (max_val - min_val)
    return weighted_threshold

def get_size_semval(size,threshold,wf=0.5):
    return 1 - dist.Normal(size - threshold, wf * jnp.sqrt(size ** 2 + threshold ** 2)).cdf(0.0)

def meaning(word, states, state_prior, color_semval = 0.95, form_semval = 0.95, k = 0.5, wf = 0.5):

    if word == "C":
        probs = jnp.where((1. == states[:,1]), color_semval, 1 - color_semval)

    if word == "F":
        probs = jnp.where((1. == states[:,2]), form_semval, 1 - form_semval)

    if word == "D":
        threshold = get_threshold_kp_sample_jax(states, state_prior, k)
        probs = jax.vmap(get_size_semval, in_axes = (0, None, None))(states, threshold, wf)[:,0] # Apply the meaning function for size adjective

    return probs
    
def incremental_literal_listener(states, color_semval = 0.95, k = 0.5):
    """
    Output: probs for each utterance
    """
    uniformStateprior = uniform_state_prior()
    # D
    probs_D = normalize(meaning("D", states, uniformStateprior, color_semval, k), axis = 0)
    # C
    probs_C = normalize(meaning("C", states, uniformStateprior, color_semval, k), axis=0)
    # F
    probs_F = normalize(meaning("F", states, uniformStateprior, color_semval, k), axis = 0)
    #CD
    probs_CD = normalize(jnp.multiply(probs_C,probs_D), axis=0)
    #CF
    probs_CF = normalize(jnp.multiply(probs_C,probs_F), axis=0)
    #DC
    probs_D_after_C = meaning("D", states, probs_C, color_semval, k)
    probs_DC = normalize(jnp.multiply(probs_D_after_C,probs_C), axis=0)
    #DF
    probs_D_after_F = meaning("D", states, probs_F, color_semval, k)
    probs_DF = normalize(jnp.multiply(probs_D_after_F,probs_F), axis=0)
    #FC
    probs_FC = probs_CF
    #FD
    probs_FD = normalize(jnp.multiply(probs_F,probs_D), axis=0)
    #DCF
    probs_D_after_CF = meaning("D", states, probs_CF, color_semval, k)
    probs_DCF = normalize(jnp.multiply(probs_D_after_CF,probs_CF), axis=0)
    #DFC
    probs_DFC = probs_DCF
    #CDF
    probs_CDF = normalize(jnp.multiply(probs_C,probs_DF), axis=0)
    #CFD
    probs_CFD = normalize(jnp.multiply(probs_C,probs_FD), axis=0)
    #FCD
    probs_FCD = probs_CFD
    #FDC
    probs_FDC = normalize(jnp.multiply(probs_F,probs_DC), axis=0)

    meaning_matrix = jnp.array([probs_D, probs_C, probs_F, probs_CD, probs_CF, probs_DC, probs_DF, probs_FC, probs_FD, probs_DCF, probs_DFC, probs_CDF, probs_CFD, probs_FCD, probs_FDC])

    return meaning_matrix

def global_speaker(states, alpha = 1, color_semval = 0.95, k = 0.5):
    """
    Output: probs for each state
    """
    current_utt_prior = jnp.log(utterance_prior(utterances))
    meaning_matrix = incremental_literal_listener(states, color_semval = color_semval, k = k)
    util_speaker = jnp.log(jnp.transpose(meaning_matrix)) + current_utt_prior
    softmax_result = jax.nn.softmax(alpha * util_speaker)
    return softmax_result



def likelihood_function_map(states, alpha, color_semval, k):
    utt_probs_conditionedReferent = global_speaker(states, alpha, color_semval, k)[0,:] # Get the probs of utterances given the first state, referent is always the first state
    with numpyro.plate("data", len(states)):
        map_predictions = numpyro.sample("obs", dist.Categorical(probs=utt_probs_conditionedReferent))
    return map_predictions

def likelihood_function(states, empirical):
    #alpha = numpyro.sample("gamma", dist.HalfNormal(5))
    alpha = 1
    color_semval = numpyro.sample("color_semvalue", dist.Uniform(0,1))
    #color_semval = 0.8
    #k = numpyro.sample("k", dist.Uniform(0, 1))
    k = 0.5
    utt_probs_conditionedReferent = global_speaker(states, alpha, color_semval, k)[0,:] # Get the probs of utterances given the first state, referent is always the first state
    with numpyro.plate("data", len(empirical)):
        numpyro.sample("obs", dist.Categorical(probs=utt_probs_conditionedReferent), obs=empirical)
    return utt_probs_conditionedReferent

def run_inference():
    states_train, empirical_train, df = import_dataset()

    # define the MCMC kernel and the number of samples
    rng_key = random.PRNGKey(11)
    rng_key, rng_key_ = random.split(rng_key)

    kernel = NUTS(likelihood_function)
    mcmc_inc = MCMC(kernel, num_warmup=10000,num_samples=10000,num_chains=1)
    mcmc_inc.run(rng_key_, states_train, empirical_train)

    # print the summary of the posterior distribution
    mcmc_inc.print_summary()

    # Get the MCMC samples and convert to a DataFrame
    posterior_inc = mcmc_inc.get_samples()
    df_inc = pd.DataFrame(posterior_inc)

    # Save the DataFrame to a CSV file
    df_inc.to_csv('../posterior_samples/production_posterior_test_4.csv', index=False)


def test_threshold():
    states_train, empirical_train, df = import_dataset()
    states_example = states_train[46]
    empirical_example = empirical_train[46]
    def uniform_state_prior(nobj=6):
        """
        Input: number of objects
        Output: list of prior probabilities for each object
        """
        prior=normalize(jnp.ones((2,nobj)))
        return prior
    stt_prior = uniform_state_prior()
    print(stt_prior)
    threshold = get_threshold_kp_sample_jax(states_example, stt_prior[0,:])
    print(threshold)

def test_threshold():
    states_manuell = jnp.array([[10., 1., 1.],
                            [10., 1., 1.],
                            [3., 1., 1.],
                            [3., 1., 0.],
                            [3., 1., 0.],
                            [1., 0., 1.]], dtype=jnp.float32)
    # stt_prior = uniform_state_prior()
    # print(stt_prior)
    # utt_prior = utterance_prior(utterances)
    # print(utt_prior)

    # threshold = get_threshold_kp_sample_jax(states_manuell, stt_prior[0,:])
    # print(threshold)
    # print(meaning("D", states_manuell, stt_prior))
    # print(meaning("C", states_manuell, stt_prior))
    # print(meaning("F", states_manuell, stt_prior))
    #print(incremental_literal_listener(states_manuell))
    #result = global_speaker(states_manuell, 1)
    #print(result)
    states_train, empirical_train, df = import_dataset()
    #result = likelihood_function(states_train)
    print(states_train[1:5])
    print(empirical_train[1:5])

if __name__ == "__main__":
    run_inference()
    #test_threshold()