import numpy as np
import pdb
import pickle as pk
import torch
from sbi import utils as utils
from sbi.inference import SNPE, SNLE, prepare_for_sbi, SNLE_A, SNRE
from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi

from sbi.inference.base import infer

import sbi_funcs as sbi_funcs
import sim_cmb_cluster_lens as sim
import settings
import precompute


#plotting
from getdist import plots, MCSamples
import matplotlib.pyplot as pl

summary_type = 'none'
param_scaling = np.array([1.0e15,1.0]) #How to scale mass and concentration parameters

#Likelihood data is used to get true observations, and for comparison
#load likelihood grid data (mock data sets, true parameter values, and corresponding likelihood grids in M200, c200 space)
likelihood_grid_filename = 'likelihood_grid_Npix32_generatefromcovTrue_usepcsFalse_1130_num5.pk'
likelihood_data = pk.load(open('./likelihood_grids/' + likelihood_grid_filename, 'rb'))
all_lnlike_mat = np.asarray(likelihood_data['lnlike_mat_list'])
M200c_arr = likelihood_data['M200c_arr']
c200c_arr = likelihood_data['c200c_arr']
num_M200c = len(M200c_arr)
num_c200c = len(c200c_arr)


settings = settings.load_settings()
N_pix = settings['N_pix']
pix_size_arcmin = settings['pix_size_arcmin']
generate_from_cov = settings['generate_from_cov']
lensing_type = settings['lensing_type']
obs_type = settings['obs_type']
z_cluster = settings['z_cluster']

#Run all the slow calculations (e.g. cosmology stuff, power spectra)
#need to prep likelihood if generating from cov
all_settings = precompute.prepare_analysis(z_cluster, prep_likelihood = True, obs_type = obs_type, N_pix = N_pix, pix_size_arcmin = pix_size_arcmin)
print("Analysis ready to go")
cluster_settings = all_settings['cluster_settings']
map_settings = all_settings['map_settings']
obs_settings = all_settings['obs_settings']
likelihood_info = all_settings['likelihood_info']

spectra = all_settings['spectra']
cosmo_params = all_settings['cosmo_params']
param_min = np.array([1.0e13, 1.0])/param_scaling
param_max = np.array([3.0e15, 10.0])/param_scaling

#Define the simulator function
def simulator_func(params):
    lensed_map, unlensed_map = sim.generate_lensed_map(params.numpy()*param_scaling, cluster_settings, map_settings, obs_settings, \
                                    spectra, cosmo_params, make_plots = False, return_unlensed = True, lensing_type = lensing_type, \
                                    generate_from_cov = generate_from_cov, likelihood_info = likelihood_info)
    return torch.from_numpy(lensed_map.flatten())

#Define the prior
prior = utils.BoxUniform(low=param_min, high=param_max)

#prepare simulator for sbi
simulator, prior = prepare_for_sbi(simulator_func, prior)
#inference = SNPE(prior=prior)
inference = SNPE(prior)

# The specific observation we want to focus the inference on.
data_index = 0
x_o = torch.from_numpy(likelihood_data['obs_map_list'][data_index].flatten()).float()
num_rounds = 2

posteriors = []
proposal = prior
#loop over rounds
for roundi in range(num_rounds):
    print("round i = ", roundi)
    theta, x = simulate_for_sbi(simulator, proposal, num_simulations=100)

    # In `SNLE` and `SNRE`, you should not pass the `proposal` to `.append_simulations()`
    density_estimator = inference.append_simulations(
        theta, x, 
    ).train()
    posterior = inference.build_posterior(density_estimator)
    posteriors.append(posterior)
    proposal = posterior.set_default_x(x_o)
