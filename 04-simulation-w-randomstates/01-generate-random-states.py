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

def main():
    #obj1 = Object(size_distribution = "flat", upper=1, lower=1, p = 0.9).generate()
    #print(obj1)
    #context1 = Context(nobj = 6,size_distribution = "flat", upper=1, lower=1, p = 0.9).generate()
    #print(context1)
    print(AllContext(sample_size= 10, nobj = 6).generate())


if __name__ == "__main__":
    main()

