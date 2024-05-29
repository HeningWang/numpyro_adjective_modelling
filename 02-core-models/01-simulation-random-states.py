# Import libraries
from dataclasses import dataclass
import torch
torch.set_default_dtype(torch.float64)  # double precision for numerical stability
import itertools
import jax
import jaxlib
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd
import numpyro
import numpyro.distributions as dist
from scipy.stats import truncnorm
from jax import random
import numpy as np
import scipy
import core_rsa
import argparse
import os

@dataclass
class Object:
    size_distribution: str = "normal"
    upper: int = 30
    lower: int = 1
    p: float = 0.5

    def sample_size(self):
        mean = (self.upper + self.lower)/2
        sd = (self.upper + self.lower)/4

        if self.size_distribution == "normal":
            a = (self.lower - mean) / sd
            b = (self.upper - mean) / sd
            return scipy.stats.truncnorm.rvs(loc = mean, scale= sd, a = a, b=b)
        
        if self.size_distribution == "left-skewed":
            mean = mean / 2 * 3
            a = (self.lower - mean) / sd
            b = (self.upper - mean) / sd
            return scipy.stats.truncnorm.rvs(loc=mean, scale=sd, a=self.lower, b=self.upper)
        
        if self.size_distribution == "right-skewed":
            mean = mean / 2
            a = (self.lower - mean) / sd
            b = (self.upper - mean) / sd
            return np.random.truncnorm(loc=mean, scale=sd, a=self.lower, b=self.upper)
        
        if self.size_distribution == "flat":
            return np.random.uniform(low=self.lower, high=self.upper)
        
        raise ValueError(f"Size distribution '{self.size_distribution}' not implemented.")

    def sample_color(self):
        return dist.Bernoulli(0.5).sample(self.key)

    def sample_form(self):
        return dist.Bernoulli(0.5).sample(self.key)

    def generate(self):
        size = self.sample_size()
        color = np.random.binomial(n=1, p=self.p)
        form = np.random.binomial(n=1, p=self.p)
        return jnp.array([size, color, form])
    
@dataclass
class Context:
    nobj: int = 6
    size_distribution: str = "normal"
    upper: int = 30
    lower: int = 1
    p: float = 0.5

    def generate(self):
        return jnp.array([
            Object(size_distribution = self.size_distribution,
                   upper = self.upper,
                   lower=self.lower,
                   p = self.p).generate() 
                   for _ in range(self.nobj)
                   ])
    

def determine_referent(context):
    """
    This method determines the index of the first row in a 2D array where:
    - the first element of the row is greater than the mean of all first elements,
    - the second and third elements are both 1.
    If no such row is found, it returns None.

    Parameters:
    array (jnp.ndarray): A 2D array where each row has at least 3 elements.

    Returns:
    int or None: The index of the first row that meets the criteria, or None if no such row is found.
    """
    # Compute the mean of the first elements of all rows
    mean = jnp.mean(context[:, 0])

    # Create a boolean array where each element is True if the corresponding row in context meets the criteria, and False otherwise
    criteria_met = jnp.where((context[:, 1] == 1) & (context[:, 2] == 1), True, False)
    print(criteria_met)
    # Find the index of the first True element in criteria_met
    index = jnp.argmax(criteria_met)

    # If no element in criteria_met is True, index will be 0, so check the first element of criteria_met to see if it's True
    if criteria_met[0]:
        return index
    else:
        return None
    
def modify_referent(context):
    """
    This method modify the attribute of the referent, which is the first row in a 2D array where:
    - the first element of the row is greater than the mean of all first elements,
    - the second and third elements are both 1.
    If no such row is found, it returns None.

    Parameters:
    array (jnp.ndarray): A 2D array where each row has at least 3 elements.

    Returns:
    array (jnp.ndarray): A 2D array where each row has at least 3 elements
    """
    modified_context = context
    # Swap the value of 0,0 with the max value of the first column
    # Find the index of max value of the first column
    max_index = jnp.argmax(context[:, 0])
    modified_context = context.at[0, 0].set(context[max_index, 0])
    modified_context = modified_context.at[max_index, 0].set(context[0, 0])
    # Set the second and third elements of the first row to 1
    modified_context = modified_context.at[0, 1].set(1)
    modified_context = modified_context.at[0, 2].set(1)

    return modified_context

def record_communicative_success(pragmatic_listener_matrix):
    """
    This function records the communicative success based on a pragmatic listener matrix.
    It returns 1 if the value at index i in the first row is greater than the value at index i in the second row.
    Otherwise, it returns 0.

    Parameters:
    pragmatic_listener_matrix (jnp.ndarray): A 2D array with 2 rows and nobj columns.
    i (int): The index of the referent.

    Returns:
    int: 1 if the value at index i in the first row is greater than the value at index i in the second row, otherwise 0.
    """
    # Check if the value at index i in the first row is greater than the value at index i in the second row
    success_array = jnp.where(pragmatic_listener_matrix[0, 0] > pragmatic_listener_matrix[1, 0], 1, 0)
    return success_array


@dataclass
class AllContext:
    sample_size: int = 10
    nobj: int = 6
    size_distribution: str = "normal"
    upper: int = 30
    lower: int = 1
    p: float = 0.5

    def generate(self):
        return jnp.array([
            Context(nobj = self.nobj,
                    size_distribution = self.size_distribution,
                   upper = self.upper,
                   lower=self.lower,
                   p = self.p).generate() 
                   for _ in range(self.sample_size)
                   ])

def simulation_run():
    # 1. first generate AllContext
    # 2. determin referent for each Context
    # 3. Apply the global speaker  
    pass

def main():
    #obj1 = Object(size_distribution = "flat", upper=1, lower=1, p = 0.9).generate()
    #print(obj1)
    #context1 = Context(nobj = 6,size_distribution = "flat", upper=1, lower=1, p = 0.9).generate()
    #print(context1)
    alpha = 1
    bias = 0
    color_semvalue = 0.95
    form_semvalue = 0.98
    wf = 0.6
    k = 0.5
    speaker = "incremental_speaker"
    world_length = 2

    states_train = AllContext(sample_size= 10, nobj = 20).generate() # this is a 3D array of shape: (sample_size, nobj, 3)
    global_speaker_vmap = jax.vmap(core_rsa.global_speaker, in_axes=(0, None))
    pragmatic_listener_vmap = jax.vmap(core_rsa.pragmatic_listener, in_axes=(0, # states
                                                                             None, # alpha
                                                                             None, # bias
                                                                             None, # color_semvalue
                                                                             None, # form_semvalue
                                                                             None, # wf
                                                                             None, # k
                                                                             None, # speaker
                                                                             None)) # world_length
    modify_referent_vmap = jax.vmap(modify_referent, in_axes=0)

    #print(states_train.shape)
    single_state = states_train[0, :, :]
    #print(single_state)
    modified_states = modify_referent_vmap(states_train)
    results = pragmatic_listener_vmap(modified_states, alpha, bias, color_semvalue, form_semvalue, wf, k, speaker, world_length)
    #print(results)
    record_communicative_success_vmap = jax.vmap(record_communicative_success, in_axes=0)
    communicative_success = record_communicative_success_vmap(results)
    print(jnp.sum(communicative_success))


if __name__ == "__main__":
    # Add argparse to the script
    parser = argparse.ArgumentParser(description='Run RSA simulation with random states.')

    # Add arguments: nobj, sample_size, color_semvalue, wf, k, speaker, alpha, bias
    parser.add_argument('--nobj', type=int, default=6, help='Number of objects in the context.')
    parser.add_argument('--sample_size', type=int, default=10, help='Number of contexts to generate.')
    parser.add_argument('--color_semvalue', type=float, default=0.95, help='Semantic value for color.')
    parser.add_argument('--form_semvalue', type=float, default=0.98, help='Semantic value for form.')
    parser.add_argument('--wf', type=float, default=0.6, help='Percerptual blur.')
    parser.add_argument('--k', type=float, default=0.5, help='k-percertage for dertermining threshold.')
    parser.add_argument('--speaker', type=str, default="incremental_speaker", help='Speaker model.')
    parser.add_argument('--alpha', type=float, default=1, help='Rationality parameter for speaker.')
    parser.add_argument('--bias', type=float, default=0, help='Bias for subjective first ordering parameter for speaker.')
    parser.add_argument('--world_length', type=int, default=2, help='World length for RSA.')
    parser.add_argument('--size_distribution', type=str, default="normal", help='Size distribution used to generate states.')
    # Parse the arguments
    args = parser.parse_args()

    # Generate the states
    states_train = AllContext(sample_size= args.sample_size, nobj = args.nobj).generate() # this is a 3D array of shape: (sample_size, nobj, 3)

    # Define a vectorized version of the global speaker, pragmatic listener, and modify_referent functions
    pragmatic_listener_vmap = jax.vmap(core_rsa.pragmatic_listener, in_axes=(0, # states
                                                                             None, # alpha
                                                                             None, # bias
                                                                             None, # color_semvalue
                                                                             None, # form_semvalue
                                                                             None, # wf
                                                                             None, # k
                                                                             None, # speaker
                                                                             None)) # world_length
    modify_referent_vmap = jax.vmap(modify_referent, in_axes=0) # The input is a 3D array, so we vectorize the function along the first axis: a single context of nobj objects
    record_communicative_success_vmap = jax.vmap(record_communicative_success, in_axes=0) # Same as above

    # Set up the referent modification ´
    modified_states = modify_referent_vmap(states_train)

    # Run the pragmatic listener
    results = pragmatic_listener_vmap(modified_states, 
                                      args.alpha, 
                                      args.bias, 
                                      args.color_semvalue, 
                                      args.form_semvalue, 
                                      args.wf, 
                                      args.k, 
                                      args.speaker, 
                                      args.world_length)
    
    # Record the communicative success
    communicative_success = record_communicative_success_vmap(results)
    sum_communicative_success = jnp.sum(communicative_success)

    # Flatten the states_train array
    communicative_success = communicative_success.tolist()
    modified_states = modified_states.tolist()
    results = results.tolist()

    # Output the results as csv
    # Create a DataFrame
    df = pd.DataFrame({
        #"array_success": communicative_success,
        "sum_success": sum_communicative_success,
        "proportion_success": sum_communicative_success/args.sample_size,
        #"full_states": states_train,
        #"modified_states": modified_states,
        #"results_pragmatic_listener": results,
        "alpha": args.alpha,
        "bias": args.bias,
        "nobj": args.nobj,
        "color_semvalue": args.color_semvalue,
        "form_semvalue": args.form_semvalue,
        "wf": args.wf,
        "k": args.k,
        "speaker": args.speaker,
        "size_distribution": args.size_distribution,
        "sample_size": args.sample_size,
        "world_length": args.world_length
    }, index = [0])

    # Save the DataFrame as a csv file
    output_filename = f"../04-simulation-w-randomstates/simulation_test_run_fewer_parameters.csv"

    # Check if the file exists
    if not os.path.isfile(output_filename):
    # If the file does not exist, create it with the header
        df.to_csv(output_filename, index=False)
    else:
        # If the file exists, append without the header
        df.to_csv(output_filename, mode='a', header=False, index=False)

