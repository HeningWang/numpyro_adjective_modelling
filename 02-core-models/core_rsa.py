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

# Mutate the dataset to include the states of the objects
# ... states are independent variables for models




# Transform/rescale slider value from range 0 to 100 to 0 to 1
# ... in order to match predicted probability from models

def transformation_data(slider_value, link = None):
    if link == "identity":
      slider_value = jnp.clip(slider_value, 0, 100)
      transformed_prob = slider_value / 100
    elif link == "logit":
        transformed_prob = 1 / (1 + math.exp(-slider_value))
    return transformed_prob

def link_function(x, param = 1):
    return 1 / (1 + jnp.exp(param * -(x - 0.5)))

def compute_alpha_beta_concentration(mu, v):
    alpha = mu * v
    beta = (1 - mu) * v
    return alpha, beta

#def Marginal(fn):
#    return memoize(lambda *args: HashingMarginal(Search(fn).run(*args)))

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

def get_results(posterior):
    results = {}
    support = posterior.enumerate_support()
    data = [posterior.log_prob(s).exp().item() for s in posterior.enumerate_support()]
    results["support"] = support
    results["probs"] = data
    return results

def normalize(arr, axis=1):
    """
    Normalize arr along axis
    """
    return arr / jnp.sum(arr, axis=axis, keepdims=True)

def get_threshold_kp(states, k=0.5):
    min_val = jnp.min(states[:,0])
    max_val = jnp.max(states[:,0])
    threshold = max_val - k * (max_val - min_val)
    return threshold

def get_threshold_kp_weighted(states, states_prior, k=0.5):
    sorted_indices = jnp.unique(jnp.argsort(states[:, 0] * states_prior), size = 6)
    sorted_states = states[sorted_indices]
    min_val = sorted_states[0, 0]
    max_val = sorted_states[-1, 0]

    weighted_threshold = max_val - k * (max_val - min_val)

    return weighted_threshold

def get_threshold_kp_sample(states, states_prior, k=0.5):
    
    xk = states[:, 0] # The values of size of all objects in a given context, shape (nobj,) 
    pk = states_prior # The prior probabilities of the objects in a given context, shape (nobj,)
    dist = scipy.stats.rv_discrete(values=(xk, pk)) # Create a discrete distribution
    sample_size = int(round(states.shape[0] / 2)) # Sample size is half of the number of objects in a given context
    samples = jnp.array(dist.rvs(size= sample_size)) # Sample from the distribution
    min_val = jnp.min(samples) # Get the minimum value of the samples
    max_val = jnp.max(samples) # Get the maximum value of the samples
    threshold = max_val - k * (max_val - min_val) # Compute the threshold

    return threshold

def get_threshold_kp_sample_jax(states, states_prior, k=0.5):
    sample_size = int(round(states.shape[0] / 2)) # Sample size is half of the number of objects in a given context
    costum_dist = dist.Categorical(probs=states_prior)
    sample_indices = jnp.unique(costum_dist.sample(random.PRNGKey(0),(1,sample_size)), size= sample_size)
    sorted_states = states[sample_indices][:,0]
    min_val = jnp.min(sorted_states)
    max_val = jnp.max(sorted_states)

    weighted_threshold = max_val - k * (max_val - min_val)
    return weighted_threshold


def adjMeaning(word, obj, current_state_prior, color_semvalue=0.98, form_semvalue=0.98, wf=0.6, k=0.5):
    colors = [1]  # Specify the color values
    sizes = [0]  # Specify the size values

    if word == 1:
        return numpyro.sample("color", numpyro.distributions.Bernoulli(color_semvalue)) if word == obj[1] else numpyro.sample("color", numpyro.distributions.Bernoulli(1 - color_semvalue))
    elif word == 0:
        threshold = get_threshold_kp(current_state_prior, k)
        size = obj[0]
        prob_big = 1 - dist.Normal(size - threshold, wf * jnp.sqrt(size ** 2 + threshold ** 2)).cdf(jnp.array([0.0]))
        return numpyro.sample("size", numpyro.distributions.Bernoulli(prob_big))
    
def get_size_semval(size,threshold,wf):
  return 1 - dist.Normal(size - threshold, wf * jnp.sqrt(size ** 2 + threshold ** 2)).cdf(0.0)

def literal_listener_one_word(states, color_semvalue = 0.98, form_semvalue = 0.98, wf = 0.6, k = 0.5):
  probs_blue = jnp.where((1. == states[:, 1]), color_semvalue, 1 - color_semvalue)
  threshold = get_threshold_kp(states, k)
  probs_big = jnp.array([1 - dist.Normal(obj[0] - threshold, wf * jnp.sqrt(obj[0] ** 2 + threshold ** 2)).cdf(0) for obj in states])
  probs = normalize(jnp.array([probs_big,probs_blue]))
  return probs

def literal_listener_recursive(word_length, states, color_semvalue = 0.90, form_semvalue = 0.98, wf = 0.6, k = 0.5, sample_based = True):
  '''
  Input: word_length: int, states: jnp.array(nobj, 3), color_semvalue: float, form_semvalue: float, wf: float, k: float
  return: jnp.array(2 * nobj) where the first row corresponds to big blue, the second row corresponds to blue big
  '''
  if word_length <= 1: # Base case
    current_states_prior = normalize(jnp.ones((2,states.shape[0]))) # Create a uniform prior of shape (2, nobj)
  else:
    current_states_prior = literal_listener_recursive(word_length - 1, states, color_semvalue = 0.98, form_semvalue = 0.98, wf = 0.6, k = 0.5)
    current_states_prior = jnp.flip(current_states_prior, axis = 0)

  probs_blue = jnp.where((1. == states[:, 1]), color_semvalue, 1 - color_semvalue) # Apply the meaning function for color adjective
  # Get the threshold for the size adjective
  if sample_based:
    threshold = get_threshold_kp_sample_jax(states, current_states_prior[0,:], k)
  else:
    threshold = get_threshold_kp(states, k)
  probs_big = jax.vmap(get_size_semval, in_axes = (0, None, None))(states[:,0], threshold, wf) # Apply the meaning function for size adjective
  probs = normalize(jnp.multiply(jnp.array([probs_big,probs_blue]), current_states_prior))
  return probs


def speaker_one_word(states, alpha = 1, bias = 0, color_semvalue = 0.98, form_semvalue = 0.98, wf = 0.6, k = 0.5):
  listener = literal_listener_one_word(states, color_semvalue, form_semvalue,wf,k)
  bias_weights = jnp.array([0, 1]) * bias
  util_speaker = jnp.log(jnp.transpose(listener)) - bias_weights
  softmax_result = jax.nn.softmax(alpha * util_speaker)
  return softmax_result

def speaker_recursive(word_length, states, alpha = 1, bias = 0, color_semvalue = 0.98, form_semvalue = 0.98, wf = 0.6, k = 0.5):
  if word_length <= 1:
    current_utt_prior = jnp.array([0, 1]) * bias # the bias is applied to the second utterance (blue big)
  else:
    current_utt_prior = speaker_recursive(word_length - 1, states, alpha, bias, color_semvalue, form_semvalue, wf, k)
    current_utt_prior = jnp.flip(current_utt_prior, axis = 1)
  listener = literal_listener_recursive(word_length, states, color_semvalue, form_semvalue, wf, k)
  #print(listener)
  util_speaker = jnp.log(jnp.transpose(listener)) - current_utt_prior
  softmax_result = jax.nn.softmax(alpha * util_speaker)
  return softmax_result

def global_speaker(states, alpha = 1, bias = 0, color_semvalue = 0.98, form_semvalue = 0.98, wf = 0.6, k = 0.5):
  listener = literal_listener_recursive(2,states, color_semvalue, form_semvalue,wf,k)
  bias_weights = jnp.array([0, 1]) * bias
  util_speaker = jnp.log(jnp.transpose(listener)) - bias_weights
  softmax_result = jax.nn.softmax(alpha * util_speaker)
  return softmax_result

def pragmatic_listener(states, alpha = 1, bias = 0, color_semvalue = 0.98, form_semvalue = 0.98, wf = 0.6, k = 0.5, speaker = "global_speaker", word_length = 2):
  prior_probs = jnp.ones((2,states.shape[0])) # Create a uniform prior of shape (2, nobj)
  if speaker == "global_speaker":
    softmax_result = global_speaker(states, alpha, bias, color_semvalue, form_semvalue, wf, k)
    # Apply Bayes' rule
    bayes_result = normalize(jnp.transpose(softmax_result) * prior_probs)
    return bayes_result
  
  if speaker == "incremental_speaker":
    softmax_result = speaker_recursive(word_length, states, alpha, bias, color_semvalue, form_semvalue, wf, k)
    # Apply Bayes' rule
    bayes_result = normalize(jnp.transpose(softmax_result) * prior_probs)
    return bayes_result
   
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

    index = 15
    states_manuell = jnp.array([[10., 1., 1.],
                    [10., 1., 1.],
                    [3., 1., 1.],
                    [3., 1., 0.],
                    [3., 1., 0.],
                    [1., 0., 1.]], dtype=jnp.float32)

    states_example = states_manuell
    #states_example = df_experiment.iloc[index, df_experiment.columns.get_loc("states")]
    condition = df_experiment.iloc[index, df_experiment.columns.get_loc("conditions")]
    distribution = df_experiment.iloc[index, df_experiment.columns.get_loc("sharpness")]
    preference = df_experiment.iloc[index, df_experiment.columns.get_loc("prefer_first_1st")]
    print(states_example)
    print(condition + " " + distribution)
    print(preference)
    print(f"literal listener one word: {literal_listener_one_word(states_example)}")
    print(f"literal listener two words: {literal_listener_recursive(2,states_example)}")
    print(f"speaker one word: {speaker_one_word(states_example)}")
    print(f"speaker two words global: {global_speaker(states_example)}")
    print("________________________________________")
    print(f"speaker two words incremental: {speaker_recursive(2,states_example)}")
    print("________________________________________")
    print(f"pragmatic listener of a global speaker: {pragmatic_listener(states_example)}")
    print("________________________________________")
    print(f"pragmatic listener of a incremental speaker: {pragmatic_listener(states_example, speaker= 'incremental_speaker')}")

def link_logit(p,s):
    x0 = 1 / (jnp.exp(1 / (-2 * s)) + 1)
    xtrans = p * (x0 - (1 - x0)) + (1 - x0)
    return s * -jnp.log((1 / xtrans) - 1) + 0.5

vectorized_speaker = jax.vmap(speaker_recursive, in_axes=(None,0,None,None,None,None,None,None))

def model_inc_utt_parallel_normal(states = None, data = None):
    gamma = numpyro.sample("gamma", dist.HalfNormal(5))
    color_semvalue = numpyro.sample("color_semvalue", dist.Uniform(0.5, 1))
    form_semvalue = color_semvalue
    k = numpyro.sample("k", dist.Uniform(0, 1))
    wf = numpyro.sample("wf", dist.Uniform(0,1))
    bias = numpyro.sample("bias", dist.HalfNormal(5))
    steepness = numpyro.sample("steepness", dist.HalfNormal(0.5))
    sigma = numpyro.sample("sigma", dist.Uniform(0,0.1))

    with numpyro.plate("data",len(states)):
        model_prob = vectorized_speaker(2, states, gamma, bias, color_semvalue, form_semvalue, wf, k)
        slider_predict = jax.vmap(link_logit, in_axes = (0,None))(model_prob[:,0,0], steepness)
        slider_predict = jnp.clip(slider_predict, 1e-5, 1 - 1e-5)
        if data is not None:
            data = jnp.clip(data, 1e-5, 1 - 1e-5)
        numpyro.sample("obs", dist.TruncatedNormal(slider_predict, sigma, low = 1e-5, high = 1 - 1e-5,), obs=data)

def run_inference():
    states_train, empirical_train = import_dataset()
    vectorized_speaker = jax.vmap(speaker_recursive, in_axes=(None,0,None,None,None,None,None,None))
    model_prob = vectorized_speaker(2,states_train, 1,1,0.5,0.5,0.5,0.5)
    print(jnp.shape(model_prob))
    print(jnp.shape(model_prob[:,0,0]))
    slider_predict = jax.vmap(link_function, in_axes = (0,None))(model_prob[:,0,0],20)

    slider_predict = jnp.clip(slider_predict, 1e-5, 1 - 1e-5)
    print(jnp.shape(slider_predict))

    # define the MCMC kernel and the number of samples
    rng_key = random.PRNGKey(11)
    rng_key, rng_key_ = random.split(rng_key)

    kernel = NUTS(model_inc_utt_parallel_normal, dense_mass=True, max_tree_depth=15, target_accept_prob=0.95)
    mcmc_inc = MCMC(kernel, num_warmup=5000,num_samples=30000,num_chains=1)
    mcmc_inc.run(rng_key_, states_train, empirical_train)

    # print the summary of the posterior distribution
    mcmc_inc.print_summary()

    # Get the MCMC samples and convert to a DataFrame
    posterior_inc = mcmc_inc.get_samples()
    df_inc = pd.DataFrame(posterior_inc)

    # Save the DataFrame to a CSV file
    df_inc.to_csv('../posterior_samples/02_inc_normal_logit_sample.csv', index=False)

if __name__ == "__main__":
    test_core_rsa()
    #run_inference()