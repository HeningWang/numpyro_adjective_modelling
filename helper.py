# import libraries
import torch
torch.set_default_dtype(torch.float64)  # double precision for numerical stability
import matplotlib.pyplot as plt
import pyro
import pyro.distributions as dist
from search_inference import HashingMarginal, memoize, Search
import math
from scipy.stats import truncnorm

# helper functions for the model
def Marginal(fn):
    return memoize(lambda *args: HashingMarginal(Search(fn).run(*args)))

def plot_dist(d, ax=None):
    support = d.enumerate_support()
    data = [d.log_prob(s).exp().item() for s in d.enumerate_support()]
    names = list(map(str, support))

    if ax is None:
        ax = plt.subplot(111)

    width = 0.3
    bins = [x-width/2 for x in range(1, len(data) + 1)]
    ax.bar(bins,data,width=width)
    ax.set_xticks(list(range(1, len(data) + 1)))
    ax.set_xticklabels(names, rotation=45, rotation_mode="anchor", ha="right")

def trunc_norm(loc, scale, lower, upper):
    x = pyro.sample("x", dist.Uniform(lower, upper))
    pyro.sample("obs", dist.Normal(loc, scale), obs=x)
    return x

def get_results(posterior):
    results = {}
    support = posterior.enumerate_support()
    data = [posterior.log_prob(s).exp().item() for s in posterior.enumerate_support()]
    results["support"] = support
    results["probs"] = data
    return results

# Mutate the dataset to include the states of the objects
# ... states are independent variables for models
def extract_states(line):
    states = []
    for i in range(6):
        new_obj = ("id", i + 1), ("size", line[4 + i]), ("color", line[10 + i]), ("form", line[16 + i])
        states.append(tuple(new_obj))
    return tuple(states)

# A linking function that links the preference ratings to the model_output(probs)
# ... the slider value is transformed to a probability value
# ... and these are dependent variables for models
# TODO: Add linking function at the output level of the model?

def transformation_data(slider_value, link = None):
    if link == "identity":
        transformed_prob = slider_value / 100
    elif link == "logit": 
        transformed_prob = 1 / (1 + math.exp(-slider_value))
    '''
    elif link == "beta":
        if slider_value == 0:
            slider_value = 1
        elif slider_value == 100 or slider_value == 101:
            slider_value = 99
        slider_value = slider_value / 100
        a = 50
        b = 50
        dist = beta(a, b)
        transformed_prob = dist.ppf(slider_value)
    '''
    return transformed_prob

def compute_alpha_beta(mean, std_dev = 0.1):
    alpha = mean * ((mean * (1 - mean) / (std_dev**2)) - 1)
    beta = (1 - mean) * ((mean * (1 - mean) / (std_dev**2)) - 1)
    return alpha, beta

