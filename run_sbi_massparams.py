import numpy as np
import pdb
import pickle as pk
import torch
from sbi import utils as utils
from sbi.inference import SNPE, prepare_for_sbi

import sbi_funcs as sbi_funcs

#plotting
from getdist import plots, MCSamples
import matplotlib.pyplot as pl

summary_type = 'none'
rescale_param = True
if rescale_param:
    rescaling = np.array([1.0e15, 1.0])
else:
    rescaling = np.array([1.0, 1.0])

#load simulation data for training SBI
data_filename = 'sims_lensingtypesimple_generatefromcovTrue_Nsims500_Npix32_1117.npz'
sim_data = np.load('./sims/' + data_filename, allow_pickle=True)
#parameter values for each simulation
params = sim_data['params']
N_sims = params.shape[0]
#generate summary statistics for all data sets
data = get_summary(sim_data['data'], summary_type = summary_type)
param_min = sim_data['param_min']/rescaling
param_max = sim_data['param_max']/rescaling

#Train the SBI model
#set up sbi priors
prior = utils.BoxUniform(low=param_min, high=param_max)

#From shivam
device = 'cpu'
neural_posterior_mean_std = utils.posterior_nn(model="maf", \
                                               hidden_features=5, num_transforms=3, device=device)

inference = SNPE(prior=prior,density_estimator=neural_posterior_mean_std)
rescaling_mat = np.tile(rescaling, (N_sims,1))
theta = torch.tensor(params/rescaling_mat, dtype=torch.float32)
x = torch.tensor(data, dtype=torch.float32)
inference.append_simulations(theta, x)
density_estimator = inference.train(max_num_epochs=1000)
posterior = inference.build_posterior(density_estimator)

#load likelihood grid data (mock data sets, true parameter values, and corresponding likelihood grids in M200, c200 space)
likelihood_grid_filename = 'likelihood_grid_Npix32_generatefromcovTrue_1117.pk'
likelihood_data = pk.load(open('./likelihood_grids/' + likelihood_grid_filename, 'rb'))
all_lnlike_mat = np.asarray(likelihood_data['lnlike_mat_list'])
M200c_arr = likelihood_data['M200c_arr']
c200c_arr = likelihood_data['c200c_arr']
num_M200c = len(M200c_arr)
num_c200c = len(c200c_arr)


#Generate plots of SBI posteriors
sbi_funcs.get_sbi_posterior_plots(likelihood_data, 'mass', posterior)