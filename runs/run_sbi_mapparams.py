import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pdb
import pickle as pk
import torch
from sbi import utils as utils
from sbi.inference import SNPE, prepare_for_sbi
from torch.distributions.normal import Normal
import matplotlib.pyplot as pl

import sbi_funcs as sbi_funcs
import jointMultivariateNormalUniform

use_pcs = True #if true, treat amplitudes of first N_pcs principal components of unlensed CMB as parameters
#otherwise, treat individual pixel values of unlensed CMB maps and mass parameters as the parameters.  Principal components
#identified from unlensed CMB.  Think this is OK, but might identify from filtered maps instead?

#likelihood data for comparing to SBI
likelihood_grid_filename = 'likelihood_grid_Npix16_generatefromcovTrue_usepcsFalse_1130_num20.pk'
if not likelihood_grid_filename is None:
    likelihood_data = pk.load(open('./likelihood_grids/' + likelihood_grid_filename, 'rb'))
else:
    likelihood_data = None

#simulation data for training SBI
data_filename = 'sims_lensingtypesimple_scaled_generatefromcovTrue_Nsims5000_Npix16_1130.npz'
sim_data = np.load('./sims/' + data_filename, allow_pickle=True)
data_obs = sim_data['data']
param_scaling = sim_data['param_scaling']
N_pix = sim_data['data_unlensed'].shape[1]**2
N_sims = sim_data['data_unlensed'].shape[0]
#mass parameter values for each simulation
mass_params = sim_data['params']
#unlensed CMB maps for each simulation
maps_unlensed = sim_data['data_unlensed'].reshape(N_sims, N_pix)
if (use_pcs):
    print("using pcs")
    N_pcs  = 50
    #Determine the PCs
    cov_unlensed = torch.from_numpy(np.cov(maps_unlensed.T)).float()
    uu, vv = np.linalg.eig(cov_unlensed)
    #For each unlensed map, determine the amplitudes of the first N_pcs principal components
    pcs = vv[:,0:N_pcs]
    pc_amplitudes = np.dot(maps_unlensed, pcs) #CHECK THIS
    
    #Should be close to diagonal
    cov_pcs = torch.from_numpy(np.cov(pc_amplitudes.T)).float()
    #Concatenate principal component amplitudes and mass parameters into single set of parameters
    all_params = np.hstack((pc_amplitudes, mass_params))
    #Set up priors
    #prior on principal component amplitudes - these should all be independent
    mean_pcs = torch.zeros(N_pcs).float()
    prior_pcs = torch.distributions.MultivariateNormal(mean_pcs,covariance_matrix = cov_pcs)
    #prior on mass parameters
    mass_param_min = sim_data['param_min']
    mass_param_max = sim_data['param_max']    
    prior_mass = utils.BoxUniform(low=mass_param_min, high=mass_param_max)
    # Form joint prior on principal component amplitudes and mass parameters
    joint_prior = jointMultivariateNormalUniform.jointMultivariateNormalUniform(prior_pcs, prior_mass)
else:
    #combine into single set of parameters (mass and unlensed CMB map)
    print("Not using PCS.  Going to be very high dimensional.")
    cov_unlensed = torch.from_numpy(np.cov(maps_unlensed.T)).float()

#Train the SBI model
#here theta are the amplitudes of pcs, and M200c and c200c, and x is the observed data
device = 'cpu'
neural_posterior = utils.posterior_nn(model="maf", hidden_features=5, num_transforms=4, device=device)
inference = SNPE(prior=joint_prior,density_estimator=neural_posterior)
theta = torch.tensor(all_params, dtype=torch.float32)
x = torch.from_numpy(data_obs.reshape(N_sims, N_pix)).float()

inference.append_simulations(theta, x)
density_estimator = inference.train(max_num_epochs=3)
posterior = inference.build_posterior(density_estimator)

#Make a bunch of plots
sbi_funcs.get_sbi_posterior_plots(likelihood_data, 'map', posterior, pcs = pcs, param_scaling = param_scaling)