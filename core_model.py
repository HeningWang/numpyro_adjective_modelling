# import libraries
import torch
torch.set_default_dtype(torch.float32)  # double precision for numerical stability

import matplotlib.pyplot as plt
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.infer import MCMC, NUTS, Predictive, TracePredictive
import pandas as pd
from siuba import *
import math
from plotnine import *
import seaborn as sns
import numpy as np
from helper import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import beta

# define lexicon
colors = ["red", "blue"]
sizes = ["big", "small"]
forms = ["square", "circle"]

# define dictionary of utterances and their costs
utterances = [
    "big blue",
    "blue big"
    ]

def get_results(posterior):
    results = {}
    support = posterior.enumerate_support()
    data = [posterior.log_prob(s).exp().item() for s in posterior.enumerate_support()]
    results["support"] = support
    results["probs"] = data
    return results
    
@Marginal
def utterance_prior(bias=torch.tensor(1)):
    probs = torch.tensor([bias,1])/(bias+1)
    n = pyro.sample("utterance_index", dist.Categorical(probs=probs))
    return utterances[n]

# relevant modules for the model
@Marginal
def perceptual_blurr(true_state, wf = 0.6):
    total = 100
    perceived_state = pyro.sample("perceived_state", dist.Categorical(probs=torch.ones(total + 1) / (total + 1)))
    pyro.factor("perceived_val", dist.Normal(true_state, wf * true_state).log_prob(perceived_state))
    return perceived_state

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
        size = obj[1][1]
        prob_big = 1 - dist.Normal(size - threshold, wf * math.sqrt(size**2 + threshold **2)).cdf(torch.tensor(0))
        #size = pyro.sample("size", perceptual_blurr(obj[1][1], wf))
        return pyro.sample("size", dist.Bernoulli(prob_big))


@Marginal
def state_prior(states):
    length = len(states)
    n = pyro.sample("state", dist.Categorical(probs=torch.ones(length) / length))
    return states[n]

# define the core models
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
def global_speaker_production(states, alpha, color_semvalue, form_semvalue, wf, k, cost_weight):
    obj = states[0] # We assume that the target object is always the first one in the list
    with poutine.scale(scale=torch.tensor(alpha)):
        utterance = pyro.sample("utterance", utterance_prior())
        pyro.factor("listener", alpha * (literal_listener(utterance,states, color_semvalue, form_semvalue, k, wf).log_prob(obj) - cost(utterance, cost_weight)))
    return  utterance


def global_speaker(states, alpha, color_semvalue = 0.98, form_semvalue = 0.98, wf = 0.6, k = 0.5, cost_weight = 0.5, sigma = 0.1):
    marginal = global_speaker_production(states, alpha, color_semvalue, form_semvalue, wf, k, cost_weight)
    results = get_results(marginal)
    prob = torch.tensor(results["probs"][0]) # the probability of the "big blue"
    a, b = compute_alpha_beta(prob, sigma)
    return dist.Beta(a, b)


lex = ["big", "blue"]

@Marginal
def lexPrior():
    probs = torch.ones(len(lex)) / len(lex)
    n = pyro.sample("lex", dist.Categorical(probs=probs))
    return lex[n]

def next_word(context):
    if not context:
        return pyro.sample("lex", lexPrior())
    else: 
        words = context.split()
        if words[-1] == "blue":
            return "big"
        elif words[-1] == "big":
            return "blue"
        else:
            return pyro.sample("lex", lexPrior())
        
@Marginal
def inc_speaker_seq_production(prefix, i, states, beta, color_semvalue, form_semvalue, wf, k):
    if i <= 1:
        current_prefix = prefix
    else:
        current_prefix = pyro.sample("seq",inc_speaker_seq_production(prefix, i-1, states, beta, color_semvalue, form_semvalue, wf, k))
    obj = states[0]
    word = next_word(current_prefix)
    if not current_prefix:
        seq = word
    else:
        seq = current_prefix + " " + word
    with poutine.scale(scale=beta):
        pyro.sample("listener", literal_listener(seq,states,color_semvalue,form_semvalue,wf,k), obs = obj)
    return seq
    
@Marginal
def inc_speaker_utt_production(states, beta, gamma, color_semvalue, form_semvalue, wf, k, cost_weight, bias):
    obj = states[0]
    with poutine.scale(scale=gamma):
        utterance = pyro.sample("utterance", utterance_prior(bias))
        pyro.factor("uttrance", inc_speaker_seq_production("",2,states, beta, color_semvalue, form_semvalue, wf, k).log_prob(utterance) - cost(utterance, cost_weight))
    return utterance
    
def inc_speaker_utt(states, beta = 1, gamma = 1, color_semvalue = 0.98, form_semvalue = 0.98, wf = 0.6, k = 0.5, cost_weight = 0.5, sigma = 0.1, bias = 2):
    posterior = inc_speaker_utt_production(states, beta, gamma, color_semvalue, form_semvalue, wf, k, cost_weight, bias)
    results = get_results(posterior)
    prob = results["probs"][0]
    a, b = compute_alpha_beta(prob, sigma)
    return dist.Normal(prob, sigma)

