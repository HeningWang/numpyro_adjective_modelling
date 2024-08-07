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
from modelling_production_data import likelihood_function, import_dataset, likelihood_function_map

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
    filename = '../posterior_samples/production_posterior_test_2.csv'
    csv_data = read_csv_to_dict(filename)
    df_inc = pd.DataFrame(csv_data)

    # Access the data dictionary
    for key, values in csv_data.items():
        print(key)
    
    states_train, empirical_train, df_experiment = import_dataset()
    color_semval = csv_data["color_semvalue"]
    # Compute map of color_semval
    color_semval_map = jnp.mean(color_semval)
    predictive = Predictive(likelihood_function, csv_data)
    predictions = predictive(rng_key_, states_train, empirical_train)["obs"]
    with numpyro.handlers.seed(rng_seed=0):
        predictions_map = likelihood_function_map(states_train, 1, color_semval_map, 0.5)
    df_pred = df_experiment
    df_pred["mean_predictions"] = jnp.mean(predictions, axis=0)
    df_pred["std_predictions"] = jnp.std(predictions, axis=0)
    df_pred["mean_predictions_map"] = predictions_map
    
    df_pred.to_csv('../posterior_samples/production_posteriorPredictive_test2.csv', index=False)

if __name__ == "__main__":
    posterior_analysis()