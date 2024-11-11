import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
from jax import random
import jax

data = jnp.array([0,1,2,2,2,3,4])

def model(data, feed_forward=False):
    
    
    if feed_forward == False:
        with numpyro.plate('data', len(data)):
            # This is the likelihood function, sample from a categorical distribution with probs as parameter, conditioned on observed data
            samples = numpyro.sample('samples', dist.Multinomial(probs=probs), obs=data)
    else:
        with numpyro.handlers.seed(rng_seed=0):
            # Define latent continuous parameters, e.g., alpha, beta
            alpha = numpyro.sample('alpha', dist.Normal(0, 1))
            beta = numpyro.sample('beta', dist.Normal(0, 1))
            gamma = numpyro.sample('gamma', dist.Normal(0, 1))
            delta = numpyro.sample('delta', dist.Normal(0, 1))
            
            # Combine the parameters into a vector
            logits = jnp.array([alpha, beta, gamma, delta])
            
            # Compute probs using a softmax function to get a probability vector over four categories
            probs = jax.nn.softmax(logits)
            samples = numpyro.sample('samples', dist.Multinomial(probs=probs))
        print(samples)

def main():
    # Set up the initial value for parameters
    initial_values = {
        'alpha': 0.1,
        'beta': 0.1,
        'gamma': 0.1,
        'delta': 0.1
    }

    # Define the NUTS sampler with an explicit initialization strategy
    nuts_kernel = NUTS(model)
    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_warmup=10, num_samples=10)

    mcmc.run(random.PRNGKey(0), data)
    mcmc.print_summary()
    samples = mcmc.get_samples()

if __name__ == "__main__":
    model(data, feed_forward=True)