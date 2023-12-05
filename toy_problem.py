#toy problem for which the likelihood of each realization is very poorly constrained
import torch
from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi
from sbi import utils as utils
import matplotlib.pyplot as pl
import pdb

import numpy as np

#x_i ~ N(mu, sigma = 10)

#prior will be more constraining than likelihood for single data set
# #prior on mu is mu in [5, 10]
mu_true = 5
min_mu = 0
max_mu = 20
sigma_true = 10.0
params_true = torch.tensor([mu_true])

def simulator_func(params):
    mu = params[0]
    data = torch.randn(1)*sigma_true + mu
    return data

prior = utils.BoxUniform(low=torch.tensor([min_mu]), high=torch.tensor([max_mu]))
simulator, prior = prepare_for_sbi(simulator_func, prior)

n_trials = 300
n_likelihood_grid = 98

#To store exact likelihood calculation and SBI calculation
lnlike_mat = np.zeros((n_trials, n_likelihood_grid))
sbi_density_mat = np.zeros((n_trials, n_likelihood_grid))
mu_array = torch.linspace(min_mu, max_mu, n_likelihood_grid)



#Train SBI model
inference = SNPE(prior = prior)
theta, x = simulate_for_sbi(simulator, proposal=prior, num_simulations=20000)
inference = inference.append_simulations(theta, x)
density_estimator = inference.train()
posterior = inference.build_posterior(density_estimator)

#Generate mock datasets and analyze using SBI and likelihood
print("generating and analyzing mock data sets")
for triali in range(n_trials):
    #generate true data set
    x_0 = simulator_func(params_true)
    #print("x_0 = ", x_0)

    #compute exact likelihood
    lnlike_exact = -(x_0 - mu_array)**2/(2.0*sigma_true**2)
    lnlike_mat[triali, :] = lnlike_exact

    #Draw SBI posterior samples
    #posterior_samples = posterior.sample((1000,), x=x_0)
    #compute SBI posterior density
    sbi_density = posterior.log_prob(mu_array[:,None], x=x_0) #None thing promotes to extra dimension
    sbi_density_mat[triali, :] = sbi_density

colors = ['blue', 'red','green', 'purple']

print("Plotting")
#stacked likelihood
fig, ax = pl.subplots(1,1)
stacked_lnlike = np.sum(lnlike_mat, axis = 0)
stacked_sbi = np.sum(sbi_density_mat, axis = 0)
ax.plot(mu_array, np.exp(stacked_lnlike - np.max(stacked_lnlike)), label = 'stacked exact likelihood', color ='dodgerblue', lw = 3)
ax.plot(mu_array, np.exp(stacked_sbi - np.max(stacked_sbi)), label = 'stacked SBI', color = 'dodgerblue', ls = 'dashed', lw = 3)
ax.plot([mu_true, mu_true], [0,1], label = 'true mu', ls = 'dashed', color = 'orange', lw = 2)
#individual likelihoods
n_plot = 2
for ploti in range(n_plot):
    ax.plot(mu_array, np.exp(lnlike_mat[ploti,:] - np.max(lnlike_mat[ploti,:])), label = 'Exact likelihood trial {}'.format(ploti), \
            color = colors[ploti], ls = 'dotted')
    ax.plot(mu_array, np.exp(sbi_density_mat[ploti,:] - np.max(sbi_density_mat[ploti,:])), label = 'SBI trial {}'.format(ploti), \
            color = colors[ploti], ls = 'dashdot')
ax.legend()
fig.savefig('./figs/toy_problem_stacked_likelihood.png')
