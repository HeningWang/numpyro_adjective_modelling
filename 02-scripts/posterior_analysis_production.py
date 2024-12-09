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
from modelling_production_data import likelihood_function, import_dataset

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
    filename = '../posterior_samples/production_posterior_test_4.csv'
    posterior_samples = read_csv_to_dict(filename)
    df_inc = pd.DataFrame(posterior_samples)
    
    states_train, empirical_train, df_experiment = import_dataset()
    color_semval = posterior_samples["color_semvalue"]
    # Compute map of color_semval
    color_semval_map = jnp.mean(color_semval)
    predictive = Predictive(likelihood_function, posterior_samples)
    predictions = predictive(rng_key_, states_train)["obs"]
    df_pred = df_experiment
    df_pred["mean_predictions"] = jnp.mean(predictions, axis=0)
    df_pred["std_predictions"] = jnp.std(predictions, axis=0)
    df_pred["predictions"] = predictions
    
    df_pred.to_csv('../posterior_samples/production_posteriorPredictive_test4_empiricalNone.csv', index=False)

if __name__ == "__main__":
    posterior_analysis()