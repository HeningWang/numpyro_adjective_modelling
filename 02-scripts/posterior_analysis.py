import jax
import jax.numpy as jnp
from jax import random, vmap
import numpy as np
import pandas as pd
import seaborn as sns
import math
import numpyro
from numpyro.diagnostics import hpdi
import numpyro.distributions as dist
from numpyro import handlers
from numpyro.infer import MCMC, NUTS
numpyro.set_platform("cpu")
import csv
from numpyro.infer import Predictive
from core_rsa import model_inc_utt_parallel_normal, import_dataset

def read_csv_to_dict(filename):
    data_dict = {}
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        for row in reader:
            for key, value in zip(header, row):
                if key in data_dict:
                    data_dict[key].append(float(value))
                else:
                    data_dict[key] = [float(value)]

    # Convert values to JAX arrays
    for key, values in data_dict.items():
        data_dict[key] = jnp.array(values)

    return data_dict

def posterior_analysis():
    rng_key = random.PRNGKey(11)
    rng_key, rng_key_ = random.split(rng_key)

    result_dict = {}
    filename = '../posterior_samples/02_inc_normal_logit_sample.csv'
    csv_data = read_csv_to_dict(filename)
    df_inc = pd.DataFrame(csv_data)

    # Access the data dictionary
    for key, values in csv_data.items():
        print(key, values[:10])

    states_train, empirical_train, df_experiment = import_dataset()
    
    predictive = Predictive(model_inc_utt_parallel_normal, csv_data)
    predictions = predictive(rng_key_, states = states_train)["obs"]

    df_pred = df_experiment
    df_pred["mean_predictions"] = jnp.mean(predictions, axis=0) - 0.5
    #df_pred["MAP_predictions"] = MAP_pred
    df_pred["prefer_first_1st"] = df_pred["prefer_first_1st"] - 0.5
    df_pred["std_predictions"] = jnp.std(predictions, axis=0)
    
    df_pred.to_csv('../posterior_samples/02_inc_normal_logit_predictions.csv', index=False)

if __name__ == "__main__":
    posterior_analysis()