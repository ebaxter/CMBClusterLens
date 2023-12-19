import numpy as np
import pdb
import pickle as pk
import torch
from sbi import utils as utils
from sbi.inference import SNPE, SNLE, prepare_for_sbi, SNLE_A, simulate_for_sbi
from sbi.utils.get_nn_models import likelihood_nn
from getdist import plots, MCSamples
import matplotlib.pyplot as pl

import sbi_funcs as sbi_funcs

summary_type = 'none'

#load simulation data for training SBI
data_filename = 'sims_lensingtypesimple_scaled_generatefromcovTrue_Nsims10000_Npix8_1205.npz'
sim_data = np.load('./sims/' + data_filename, allow_pickle=True)
#parameter values for each simulation
params = sim_data['params']
N_sims = params.shape[0]
#generate summary statistics for all data sets
data = sbi_funcs.get_summary(sim_data['data'], summary_type = summary_type)
param_min = sim_data['param_min']
param_max = sim_data['param_max']
param_scaling = sim_data['param_scaling']

prior = utils.BoxUniform(low=param_min, high=param_max)

device = 'cpu'
neural_likelihood =likelihood_nn(model="maf", num_transforms=3, device=device)



inference = SNLE(prior=prior, density_estimator=neural_likelihood)
#data for training
x = torch.tensor(data, dtype=torch.float32)
#parameters corresponding to each simulation
theta = torch.tensor(sim_data['params'], dtype=torch.float32)
inference.append_simulations(theta, x)
density_estimator = inference.train(max_num_epochs=1000)
posterior = inference.build_posterior(density_estimator)

#Compare to likelihood calculation
#load likelihood grid data (mock data sets, true parameter values, and corresponding likelihood grids in M200, c200 space)
likelihood_grid_filename = 'likelihood_grid_Npix8_generatefromcovTrue_usepcsFalse_1205_num400.pk'
likelihood_data = pk.load(open('./likelihood_grids/' + likelihood_grid_filename, 'rb'))
true_lnlike_mat = np.asarray(likelihood_data['lnlike_mat_list'])
snle_lnlike_mat = np.zeros(true_lnlike_mat.shape)
M200c_arr = likelihood_data['M200c_arr']
c200c_arr = likelihood_data['c200c_arr']
num_M200c = len(M200c_arr)
num_c200c = len(c200c_arr)

#For 2D grid
M200c_mat, c200c_mat = np.meshgrid(M200c_arr/param_scaling[0], c200c_arr/param_scaling[1], indexing = 'ij')
param_grid = torch.from_numpy(np.vstack((M200c_mat.flatten(), c200c_mat.flatten())).T).float()
ntrials = len(likelihood_data['obs_map_list'])
for triali in range(0, ntrials):
    #observed data for trial i
    x_0 = likelihood_data['obs_map_list'][triali].flatten()
    #Get likelihood
    snle_density = posterior.log_prob(param_grid, x=x_0) 
    #put into matrix
    snle_lnlike_mat[triali, :,:] = snle_density.reshape((num_M200c, num_c200c))

#Plot results
true_M200c = likelihood_data['true_params'][0]
true_c200c = likelihood_data['true_params'][1]
stacked_true_lnlike = np.sum(true_lnlike_mat, axis = 0)
stacked_snle_lnlike = np.sum(snle_lnlike_mat, axis = 0)
fig, ax = pl.subplots(2,1, figsize = (6,12))
ax[0].contourf(c200c_arr, M200c_arr, stacked_true_lnlike)
ax[1].contourf(c200c_arr, M200c_arr, stacked_snle_lnlike )
ax[0].plot(true_c200c, true_M200c, 'kx', markersize = 10)
ax[1].plot(true_c200c, true_M200c, 'kx', markersize = 10)
ax[0].set_xlabel('c200c')
ax[1].set_ylabel('M200c')
ax[0].set_title('stacked true lnlike')
ax[1].set_title('stacked SNLE lnlike')
fig.tight_layout()
fig.savefig('./figs/SNLE_massparam_contours.png')




fig, ax = pl.subplots(2,1, figsize = (6,12))
triali = 0
ax[0].contourf(c200c_arr, M200c_arr, true_lnlike_mat[triali, :,:] - true_lnlike_mat[triali, :,:].max(), levels = [-5., -2., -1., 0.])
ax[1].contourf(c200c_arr, M200c_arr, snle_lnlike_mat[triali, :,:] - snle_lnlike_mat[triali, :,:].max(), levels = [-5., -2., -1., 0.])
ax[0].plot(true_c200c, true_M200c, 'kx', markersize = 10)
ax[1].plot(true_c200c, true_M200c, 'kx', markersize = 10)
ax[0].set_xlabel('c200c')
ax[1].set_ylabel('M200c')
ax[0].set_title('indiv true lnlike')
ax[1].set_title('indiv SNLE lnlike')
fig.tight_layout()
fig.savefig('./figs/SNLE_massparam_contours_indiv.png')

pdb.set_trace()