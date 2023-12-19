#toy problem for which the likelihood of each realization is very poorly constrained
import torch
from sbi.inference import SNPE, SNLE, SNRE, prepare_for_sbi, simulate_for_sbi
from sbi import utils as utils
import matplotlib.pyplot as pl
import pdb
import numpy as np

###
#See https://github.com/mlcolab/sbi-workshop/blob/main/slides/2_3_snle_snre.ipynb


#x_i ~ N(mu, sigma)

#prior will be more constraining than likelihood for single data set
# #prior on mu is mu in [5, 20]
mu_true = 15
min_mu = 0
max_mu = 50
sigma_true = 30.0
params_true = torch.tensor([mu_true])

def simulator_func(params):
    mu = params[0]
    data = torch.randn(1)*sigma_true + mu
    return data

prior = utils.BoxUniform(low=torch.tensor([min_mu]), high=torch.tensor([max_mu]))
simulator, prior = prepare_for_sbi(simulator_func, prior)

n_trials = 100
n_likelihood_grid = 50

#To store exact likelihood calculation and SBI calculation
lnlike_mat = np.zeros((n_trials, n_likelihood_grid))
snpe_density_mat = np.zeros((n_trials, n_likelihood_grid))
snle_density_mat = np.zeros((n_trials, n_likelihood_grid))
snre_density_mat = np.zeros((n_trials, n_likelihood_grid))

mu_array = torch.linspace(min_mu, max_mu, n_likelihood_grid)

#Train models
#data for training
theta, x = simulate_for_sbi(simulator, proposal=prior, num_simulations=2000)

#training SNPE
inference_snpe = SNPE(prior = prior)
inference_snpe = inference_snpe.append_simulations(theta, x)
density_estimator_snpe = inference_snpe.train()
posterior_snpe = inference_snpe.build_posterior(density_estimator_snpe)
#training SNLE
inference_snle = SNLE(prior = prior)
inference_snle = inference_snle.append_simulations(theta, x)
density_estimator_snle = inference_snle.train()
posterior_snle = inference_snle.build_posterior(density_estimator_snle)
#training SNRE
inference_snre = SNRE(prior = prior)
inference_snre = inference_snre.append_simulations(theta, x)
density_estimator_snre = inference_snre.train()
posterior_snre = inference_snre.build_posterior(density_estimator_snre)

#Generate mock datasets and analyze using SBI and likelihood
print("generating and analyzing mock data sets")
for triali in range(n_trials):
    #generate true data set
    x_0 = simulator_func(params_true)
    #compute exact likelihood
    lnlike_exact = -(x_0 - mu_array)**2/(2.0*sigma_true**2)
    lnlike_mat[triali, :] = lnlike_exact

    #compute SBI posteriors or likelihood
    snpe_density = posterior_snpe.log_prob(mu_array[:,None], x=x_0)  #None thing promotes to extra dimension
    snpe_density_mat[triali, :] = snpe_density
    #Is this returning likelihood????
    snle_density = posterior_snle.log_prob(mu_array[:,None], x=x_0)
    snle_density_mat[triali, :] = snle_density
    snre_density = posterior_snre.log_prob(mu_array[:,None], x=x_0)
    snre_density_mat[triali, :] = snre_density



print("Plotting")
colors = ['blue', 'red','green', 'purple']
#stacked likelihood
fig, ax = pl.subplots(1,1)
stacked_lnlike = np.sum(lnlike_mat, axis = 0)
stacked_snpe = np.sum(snpe_density_mat, axis = 0)
stacked_snle = np.sum(snle_density_mat, axis = 0)
stacked_snre = np.sum(snre_density_mat, axis = 0)

ax.plot(mu_array, np.exp(stacked_lnlike - np.max(stacked_lnlike)), label = 'stacked exact likelihood', color ='dodgerblue', lw = 3)
ax.plot(mu_array, np.exp(stacked_snpe - np.max(stacked_snpe)), label = 'stacked SNPE', color = 'forestgreen', lw = 3, ls = 'dotted')
ax.plot(mu_array, np.exp(stacked_snle - np.max(stacked_snle)), label = 'stacked SNLE', color = 'orangered', lw = 3, ls = 'dashed')
ax.plot(mu_array, np.exp(stacked_snre - np.max(stacked_snre)), label = 'stacked SNRE', color = 'purple', lw = 3)
ax.plot([mu_true, mu_true], [0,1], label = 'true mu', ls = 'dashed', color = 'black', lw = 2)
#individual likelihoods
#n_plot = 2
#for ploti in range(n_plot):
#    ax.plot(mu_array, np.exp(lnlike_mat[ploti,:] - np.max(lnlike_mat[ploti,:])), label = 'Exact likelihood trial {}'.format(ploti), \
#            color = colors[ploti], ls = 'dotted')
#    ax.plot(mu_array, np.exp(sbi_density_mat[ploti,:] - np.max(sbi_density_mat[ploti,:])), label = 'SBI trial {}'.format(ploti), \
#            color = colors[ploti], ls = 'dashdot')
ax.legend()
fig.savefig('./figs/toy_problem_stacked_likelihood.png')
