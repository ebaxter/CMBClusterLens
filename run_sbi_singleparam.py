import numpy as np
import pdb
import pickle as pk
import torch
from sbi import utils as utils
from sbi.inference import SNPE, SNLE, prepare_for_sbi, SNLE_A, simulate_for_sbi, SNRE
from getdist import plots, MCSamples
import matplotlib.pyplot as pl

from sbi.utils.get_nn_models import likelihood_nn, posterior_nn

#cluster lensing functions
import sbi_funcs as sbi_funcs
import likelihood_funcs
import sim_cmb_cluster_lens as sim
import precompute
import settings
N_pix = 6
summary_type = 'none'
device = 'cpu'
num_sims = 5000
method = 'SNRE'

#'truth' values
c200c_default = 4.0
M200c_default = 3.0e15
settings = settings.load_settings()
settings['N_pix'] = N_pix
pix_size_arcmin = settings['pix_size_arcmin']
generate_from_cov = settings['generate_from_cov']
lensing_type = settings['lensing_type']
obs_type = settings['obs_type']
z_cluster = settings['z_cluster']

#Run all the slow calculations (e.g. cosmology stuff, power spectra)
#need prep likelihood if generating from cov
all_settings = precompute.prepare_analysis(z_cluster, prep_likelihood = True, obs_type = obs_type, N_pix = N_pix, pix_size_arcmin = pix_size_arcmin)
print("Analysis ready to go")
cluster_settings = all_settings['cluster_settings']
map_settings = all_settings['map_settings']
obs_settings = all_settings['obs_settings']
likelihood_info = all_settings['likelihood_info']

spectra = all_settings['spectra']
cosmo_params = all_settings['cosmo_params']

def simulator_func(params):
    M200c = params[0]*1.0e15
    c200c = c200c_default
    new_params = np.array([M200c, c200c])
    #This uses numpy arrays, not torch tensors
    lensed_map, unlensed_map = sim.generate_lensed_map(new_params, cluster_settings, map_settings, obs_settings, \
                                        spectra, cosmo_params, make_plots = False, return_unlensed = True, lensing_type = lensing_type, \
                                            generate_from_cov = generate_from_cov, likelihood_info = likelihood_info)
    #convert back to torch for sbi
    return torch.from_numpy(lensed_map.flatten()).float()

min_M200c = 0.1e15
max_M200c = 10.0e15
low = torch.tensor([min_M200c/1.0e15], device = device)
high = torch.tensor([max_M200c/1.0e15], device = device)
prior = utils.BoxUniform(low=low, high=high)
simulator, prior = prepare_for_sbi(simulator_func, prior)

#Initialize the neural network model and inference object
if method == 'SNLE':
    neural_likelihood =likelihood_nn(model = 'maf', num_transforms=10, device=device)
    inference = SNLE(prior=prior, density_estimator=neural_likelihood)
elif method == 'SNRE':
    #neural_posterior = posterior_nn(model="nsf", hidden_features=60, num_transforms=3)    
    inference = SNRE(prior=prior, classifier="resnet")
elif method == 'SNPE':
    neural_posterior = posterior_nn(model="nsf", hidden_features=60, num_transforms=3)
    inference = SNPE(prior=prior, density_estimator=neural_posterior)
#generate simulations
theta, x = simulate_for_sbi(simulator, proposal=prior, num_simulations=num_sims)

#add the simulations to the inference object
inference.append_simulations(theta, x)


density_estimator = inference.train(max_num_epochs=1000)

#build the posterior
posterior = inference.build_posterior(density_estimator)

#Compare to likelihood calculation
num_M200c = 50 #number of M200c values to evaluate
M200c_arr = torch.linspace(min_M200c, max_M200c, num_M200c)
N_trials = 10
true_lnlike_mat = np.zeros((N_trials, num_M200c))
sbi_lnlike_mat = np.zeros((N_trials, num_M200c))
for triali in range(0, N_trials):
    #Generate a mock data set with default mass
    x_0 = simulator_func(torch.tensor([M200c_default/1.0e15]).float())
    #Analyze using exact likelihood
    #Get likelihood
    for mi in range(0, len(M200c_arr)):
        params = np.array([M200c_arr[mi], c200c_default])
        use_pcs = False
        lnlike, term1, term2 = likelihood_funcs.lnlikelihood(params, cluster_settings, map_settings, obs_settings, spectra, \
                            cosmo_params, likelihood_info, x_0, use_pcs = use_pcs)
        true_lnlike_mat[triali, mi] = lnlike
    #Get SBI estimate of likelihood
    sbi_density = posterior.log_prob(M200c_arr[:,None]/1.0e15, x=x_0) 
    #put into matrix
    sbi_lnlike_mat[triali, :,] = sbi_density

#Plot results
stacked_sbi_lnlike = np.sum(sbi_lnlike_mat, axis = 0)
stacked_sbi_lnlike = stacked_sbi_lnlike - np.max(stacked_sbi_lnlike)
stacked_true_lnlike = np.sum(true_lnlike_mat, axis = 0)
stacked_true_lnlike = stacked_true_lnlike - np.max(stacked_true_lnlike)
fig, ax = pl.subplots(1,1, figsize = (6,6))
ax.plot(M200c_arr, np.exp(stacked_true_lnlike), label = 'True likelihood', lw = 3)
ax.plot(M200c_arr, np.exp(stacked_sbi_lnlike), label = 'SBI, method = ' + method, lw = 3)
ax.plot([M200c_default, M200c_default], [0., 1.], label = 'True mass')
ax.set_xlabel('M200c')
ax.set_ylabel('lnlike')
ax.legend()
fig.savefig('./figs/SBI_massonly_Npix{}_Nsims{}_method{}.png'.format(N_pix, num_sims, method))

