import core_rsa

import torch
torch.set_default_dtype(torch.float64)  # double precision for numerical stability

import matplotlib.pyplot as plt
import itertools
import time
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from search_inference import HashingMarginal, memoize, Search, BestFirstSearch
from pyro.infer import MCMC, NUTS, HMC, EmpiricalMarginal, Importance, SVI, Trace_ELBO, TraceEnum_ELBO, config_enumerate, Predictive
from pyro.poutine.trace_messenger import TraceMessenger
import pandas as pd
from siuba import *
import math
from plotnine import *
import seaborn as sns
import numpy as np
from helper import Marginal, plot_dist, get_results
from sklearn.model_selection import train_test_split
from scipy.stats import truncnorm
import copy

def import_dataset(file_path = "../01-dataset/01-production-data-preprocessed.csv"):

    def encode_states(line):
        states = []
        for i in range(6):
            new_obj = ("id", i + 1), ("size", line[6 + i]), ("color", line[12 + i]), ("form", line[18 + i])
            states.append(tuple(new_obj))
        return tuple(states)
    
   # Import the data
    df = pd.read_csv(file_path)

    # Mutate the dataset to include the states of the objects
    df["states"] = df.apply(lambda row: encode_states(row), axis=1)

    # split the dataset into training and test sets
    #train, test = train_test_split(df, test_size=0.99, random_state=42)
    # Use the whole dataset as training set
    train = df
    states_train = train.states
    empirical_train = train.annotation

    return states_train, empirical_train, df

# define lexicon
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

@Marginal
def utterance_prior():
    utterances = [
    "big",
    "blue",
    "circle",
    "blue big", #CD
    "blue circle",
    "big blue",
    "big circle",
    "circle blue",
    "circle big", #FD
    "big blue circle",
    "big circle blue",
    "blue big circle",
    "blue circle big", # CFD
    "circle blue big", # FCD
    "circle big blue", # FDC
    ]
    length = len(utterances)
    n = pyro.sample("utterance", dist.Categorical(probs=torch.ones(length) / length))
    return utterances[n]

@Marginal
def state_prior(states):
    length = len(states)
    n = pyro.sample("state", dist.Categorical(probs=torch.ones(length) / length))
    return states[n]

# relevant modules for the model
def perceptual_blurr(true_state, weber_fraction = 0.6): # TODO: Add this parameter to the model inference?
    # Empirically determined by simulations of random states, this is the value that gives the desired output of global speaker model (slightly prefer the conventional ordering)
    # if weber_fraction is a tensor, we need to convert it to a float
    if torch.is_tensor(weber_fraction):
        weber_fraction = weber_fraction.clone().detach().numpy()
    loc = true_state
    scale = weber_fraction * true_state
    clip_a = 1
    clip_b = 15
    a, b = (clip_a - loc) / scale, (clip_b - loc) / scale
    blur = round(truncnorm(a = a, b = b, loc = true_state, scale = scale).rvs())
    return blur

def get_threshold_kp(current_state_prior, k = 0.5):
    objs = current_state_prior.enumerate_support()
    measures_array = sorted([x[1] for obj in objs for x in obj if x[0] == 'size'])
    min_val = measures_array[0]
    max_val = measures_array[-1]
    threshold = max_val - k * (max_val - min_val)
    return threshold


def cost(utterance, color_cost=1, size_cost=1, form_cost=1, costWeight=0.5): # TODO: Add those parameter to the model inference?
    if len(utterance.split()) <= 1:
        cost = size_cost if utterance in sizes else color_cost if utterance in colors else form_cost
        return cost * costWeight
    else:
        words = utterance.split()
        cost = sum([size_cost if x in sizes else color_cost if x in colors else form_cost for x in words])
        return cost * costWeight


def adjMeaning(word, obj, current_state_prior, color_semvalue=0.98, form_semvalue=0.98, wf=0.6, k=0.5):
    # Define meaning function
    if word in colors:
        return pyro.sample("color", dist.Bernoulli(color_semvalue)) if word == obj[2][1] else pyro.sample("color", dist.Bernoulli(1 - color_semvalue))
    elif word in forms:
        return pyro.sample("form", dist.Bernoulli(form_semvalue)) if word == obj[3][1] else pyro.sample("form", dist.Bernoulli(1 - form_semvalue))
    elif word in sizes:
        threshold = get_threshold_kp(current_state_prior, k)
        size = perceptual_blurr(true_state=obj[1][1], weber_fraction=wf)
        return size >= threshold
    
@Marginal
def literal_listener(words, states, color_semvalue = 0.98, form_semvalue = 0.98, wf = 0.6, k = 0.5,):
    if len(words.split()) <= 1:
        current_state_prior = state_prior(states)
        current_word = words
    else:
        current_state_prior = literal_listener(words.split()[1:][0], states)
        current_word = words.split()[0]
    obj = pyro.sample("obj", current_state_prior)
    utt_truth_val = adjMeaning(current_word, obj, current_state_prior, color_semvalue, form_semvalue, wf, k)
    pyro.factor("literal_meaning", 0. if utt_truth_val == True else -9999999.)
    return obj

@Marginal
def global_speaker(states, alpha = 1, color_semvalue = 0.98, form_semvalue = 0.98, wf = 0.6, k = 0.5, cost_weight = 0.5):
    obj = states[0] # the cache function of Python only works for immutable objects
    with poutine.scale(scale=alpha):
        utterance = pyro.sample("utterance", utterance_prior())
        pyro.factor("listener", literal_listener(utterance,states, color_semvalue, form_semvalue, k, wf).log_prob(obj) - cost(utterance, cost_weight))
        categorie = classify_utterance(utterance)
    return  categorie

def condition2(states, data):
    alpha = pyro.sample(f"alpha", dist.Uniform(0, 30))
    color_semvalue = pyro.sample(f"color_semvalue", dist.Beta(40, 2))
    form_semvalue = pyro.sample(f"form_semvalue", dist.Beta(40, 2))
    k = pyro.sample(f"k", dist.Uniform(0, 1))
    cost_weight = pyro.sample(f"cost_weight", dist.Uniform(0, 1))
    wf = pyro.sample(f"wf", dist.Uniform(0, 1))
    for i in pyro.plate("data_loop", len(states)):
        pyro.sample(f"obs_{i}", global_speaker(states[i], alpha, color_semvalue, form_semvalue, wf, k, cost_weight), obs=data[i])

def run_inference(states, empirical, numSamples = 1000, warmupSteps = 1000):
    kernel = NUTS(condition2)
    mcmc = MCMC(kernel, num_samples=numSamples, warmup_steps=warmupSteps)
    mcmc.run(states, empirical)
    return mcmc
def test():
    # Import the dataset
    states_train, empirical_train, df = import_dataset()
    states_example = states_train[46]
    empirical_example = empirical_train[46]
    print(empirical_example)
    print(states_example)
    #l = literal_listener("big blue", states_example)
    g = global_speaker(states_example)
    print(get_results(g))
    mcmc = run_inference(states_train, empirical_train)
    mcmc.summary()
    samples = mcmc.get_samples()
    df = pd.DataFrame(samples)

    # Save the DataFrame to a CSV file
    df.to_csv('posterior_production_gb.csv', index=False)

if __name__ == "__main__":
    test()