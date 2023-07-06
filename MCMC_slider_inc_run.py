from core_model import *
from pyro.infer import MCMC, NUTS, HMC
import tqdm
from pyro.infer import Predictive
import torch
torch.set_default_dtype(torch.float32)

def split_df(df,split=False, test_size=0.99, random_state=42):
    if split:
        train, test = train_test_split(df, test_size=test_size, random_state=random_state)
        train.reset_index(inplace=True, drop=True)
        test.reset_index(inplace=True, drop=True)
        return train
    else:
        return df
    
def link_function(x, link = "identity", param = None):
    if link == "identity":
        return x
    elif link == "logit":
        return np.log(x/(1-x))
    elif link == "rapidlogit":
        # rescale parameter of steepness to be between 0 and 1, and with 0.1 nearly identity
        # param = 50 * param
        return 1 / (1 + torch.exp(param * -(x - 0.5)))
    
# define the conditioned model for MCMC
def model_inc_utt_serial(states, data):
    beta = 1
    gamma = pyro.sample("gamma", dist.HalfNormal(5))
    color_semvalue = pyro.sample("color_semvalue", dist.Uniform(0, 1))
    form_semvalue = color_semvalue
    k = pyro.sample("k", dist.Uniform(0, 1))
    cost_weight = 0.5
    wf = 0.5
    sigma = pyro.sample("sigma", dist.HalfNormal(0.5))
    bias = pyro.sample("bias", dist.HalfNormal(5)) + 1
    stepness = pyro.sample("stepness", dist.HalfNormal(1))
    for i in pyro.plate("data_loop", len(data)):
        model_prob = get_results(inc_speaker_utt_production(states[i], beta, gamma, color_semvalue, form_semvalue,wf,k,cost_weight, bias))["probs"][0]
        slider_predict = link_function(model_prob, link = "rapidlogit", param = stepness) 
        #pyro.sample("obs_{}".format(i), dist.Normal(slider_predict,sigma), obs=data[i])
        pyro.sample("obs_{}".format(i), dist.Normal(slider_predict,sigma))

def compute_alpha_beta_concentration(mu, v):
    alpha = mu * v
    beta = (1 - mu) * v
    return alpha, beta

# define the conditioned model for MCMC
def model_inc_utt_serial_beta(states, data):
    beta = 1
    gamma = pyro.sample("gamma", dist.HalfNormal(5))
    color_semvalue = pyro.sample("color_semvalue", dist.Uniform(0, 1))
    form_semvalue = color_semvalue
    k = pyro.sample("k", dist.Uniform(0, 1))
    cost_weight = 0.5
    wf = 0.5
    #sigma = pyro.sample("sigma", dist.HalfNormal(0.5))
    bias = pyro.sample("bias", dist.HalfNormal(5)) + 1
    stepness = pyro.sample("stepness", dist.HalfNormal(0.5))
    v = pyro.sample("v", dist.Uniform(1e-5,5))
    for i in pyro.plate("data_loop", len(data)):
        model_prob = get_results(inc_speaker_utt_production(states[i], beta, gamma, color_semvalue, form_semvalue,wf,k,cost_weight, bias))["probs"][0]
        slider_predict = link_function(model_prob, link = "rapidlogit", param = stepness) 
        slider_predict = torch.clamp(slider_predict, 1e-5, 1-1e-5)
        obs = torch.clamp(data[i], 1e-5, 1-1e-5)
        alpha, beta = compute_alpha_beta_concentration(slider_predict, v)
        pyro.sample("obs_{}".format(i), dist.Beta(alpha,beta), obs=obs) # use this for inference
        #pyro.sample("obs_{}".format(i), dist.Beta(alpha,beta)) # use this for prior predictive

# Import dataset
df = pd.read_csv('code/pyro_implementation/dataset/dataset_slider.csv')
# subset data to only include combination dimension_color
df = df[df['combination'] == 'dimension_color']
df.reset_index(inplace=True, drop=True)

# Mutate the dataset to include the states of the objects
df_experiment = df.copy()
df_experiment["states"] = df_experiment.apply(lambda row: extract_states(row), axis=1)
df_experiment.prefer_first_1st = df_experiment.prefer_first_1st.apply(lambda x: transformation_data(x, link = "identity"))

# split data into train and test set
#train = split_df(df_experiment, split=True, test_size=0.99, random_state=42)
train = df_experiment
states_train = tuple(train.iloc[:, train.columns.get_loc("states")])
empirical_train = torch.tensor(train.iloc[:, train.columns.get_loc("prefer_first_1st")])

def main():
    # define the MCMC kernel and the number of samples
    kernel = NUTS(model_inc_utt_serial_beta, target_accept_prob=0.5)
    mcmc_inc = MCMC(kernel, num_samples=250, warmup_steps=100, num_chains=4)
    mcmc_inc.run(states_train, empirical_train)

    # print the summary of the posterior distribution
    mcmc_inc.summary()
    mcmc_inc.diagnostics()

    # Get the MCMC samples and convert to a DataFrame
    posterior_inc = mcmc_inc.get_samples()
    df_inc = pd.DataFrame(posterior_inc)

    # Save the DataFrame to a CSV file
    df_inc.to_csv('posterior_inc_utt_slider.csv', index=False) 

def prior_predictive_normal():
    num_samples = 10
    prior_samples = {
        "gamma": torch.stack([dist.HalfNormal(5).sample()  for _ in range(num_samples)]),
        "color_semvalue": torch.stack([dist.Uniform(0, 1).sample() for _ in range(num_samples)]),
        "k": torch.stack([dist.Uniform(0, 1).sample() for _ in range(num_samples)]),
        "sigma": torch.stack([dist.HalfNormal(0.5).sample() for _ in range(num_samples)]),
        "bias": torch.stack([dist.HalfNormal(5).sample() +1 for _ in range(num_samples)]),
        "stepness": torch.stack([dist.HalfNormal(5).sample() + 1  for _ in range(num_samples)]),
    }
    predictive = Predictive(model_inc_utt_serial, prior_samples, num_samples=num_samples)
    prior_predictive = predictive(states_train, empirical_train)
    print(prior_predictive)
    #  Save prior predictive samples to dataframe
    df_inc_prior = pd.DataFrame(prior_predictive)

def prior_predictive_beta():
    num_samples = 10
    prior_samples = {
        "gamma": torch.stack([dist.HalfNormal(5).sample()  for _ in range(num_samples)]),
        "color_semvalue": torch.stack([dist.Uniform(0, 1).sample() for _ in range(num_samples)]),
        "k": torch.stack([dist.Uniform(0, 1).sample() for _ in range(num_samples)]),
        "thete_v": torch.stack([dist.Uniform(0,2).sample() for _ in range(num_samples)]),
        "bias": torch.stack([dist.HalfNormal(5).sample() +1 for _ in range(num_samples)]),
        "stepness": torch.stack([dist.HalfNormal(5).sample() + 1  for _ in range(num_samples)]),
    }
    predictive = Predictive(model_inc_utt_serial_beta, prior_samples, num_samples=num_samples)
    prior_predictive = predictive(states_train, empirical_train)
    print(prior_predictive)
    #  Save prior predictive samples to dataframe
    df_inc_prior = pd.DataFrame(prior_predictive)

    # Save the DataFrame to a CSV file
    df_inc_prior.to_csv('prior_predictive_inc_utt_slider.csv', index=False) 



if __name__ == '__main__':
    main()
    #prior_predictive_beta()
    #prior_predictive_normal()