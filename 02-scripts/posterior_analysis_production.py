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
from modeling_production_improved import likelihood_function_global_speaker, import_dataset

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
    # Set the random seed
    rng_key = random.PRNGKey(11)
    rng_key, rng_key_ = random.split(rng_key)

    # Import the posterior samples
    result_dict = {}
    filename = '../posterior_samples/production_posterior_full_gb1.csv'
    posterior_samples = read_csv_to_dict(filename)
    df_inc = pd.DataFrame(posterior_samples)
    
    # Import the dataset
    data = import_dataset()
    states_train = data["states_train"]
    empirical_train_seq = data["empirical_seq"]
    empirical_train_flat = data["empirical_flat"]
    empirical_train_seq_flat = data["empirical_seq_flat"]
    df_experiment = data["df"]

    # Define the predictive model
    predictive = Predictive(likelihood_function_global_speaker, posterior_samples)
    predictions = predictive(rng_key_, states_train)["obs"]
    
    # Save the predictions to a CSV file
    # Transpose to (n_rows, n_samples)
    predictions_t = predictions.T  # jnp.transpose(predictions)

    # Convert to Python lists of floats (optional: jnp → np)
    pred_list = np.array(predictions_t).tolist()  # now a list of lists

    # Add sample-wise predictions into DataFrame
    df_pred = df_experiment.copy()
    df_pred["predictions"] = pred_list
    df_pred["mean_predictions"] = jnp.mean(predictions, axis=0)
    df_pred["std_predictions"] = jnp.std(predictions, axis=0)
        
    df_pred.to_csv('../posterior_samples/production_posteriorPredictive_full_gb1.csv', index=False)

if __name__ == "__main__":
    posterior_analysis()