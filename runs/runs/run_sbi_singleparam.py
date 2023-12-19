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
device = 'cpu'
num_sims = 10000
method = 'SNRE'

#'truth' values
c200c_default = 4.0
M200c_default = 3.0e15
settings = settings.load_settings()
settings['N_pix'] = N_pix
pix_size_arcmin = settings['pix_size_arcmin']
generate_from_cov = settings['generate_from_cov']

lensing_type_train = 'simple'
lensing_type_test = 'full'

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

def simulator_func_train(params):
    M200c = params[0]*1.0e15
    c200c = c200c_default
    new_params = np.array([M200c, c200c])
    #This uses numpy arrays, not torch tensors
    lensed_map, unlensed_map = sim.generate_lensed_map(new_params, cluster_settings, map_settings, obs_settings, \
                                        spectra, cosmo_params, make_plots = False, return_unlensed = True, lensing_type = lensing_type_train, \
                                            generate_from_cov = generate_from_cov, likelihood_info = likelihood_info)
    #convert to torch for sbi
    return torch.from_numpy(lensed_map.flatten()).float()

def simulator_func_test(params):
    M200c = params[0]*1.0e15
    c200c = c200c_default
    new_params = np.array([M200c, c200c])
    #This uses numpy arrays, not torch tensors
    lensed_map, unlensed_map = sim.generate_lensed_map(new_params, cluster_settings, map_settings, obs_settings, \
                                        spectra, cosmo_params, make_plots = False, return_unlensed = True, lensing_type = lensing_type_test, \
                                            generate_from_cov = generate_from_cov, likelihood_info = likelihood_info)
    #convert to torch for sbi
    return torch.from_numpy(lensed_map.flatten()).float()

min_M200c = 0.1e15
max_M200c = 10.0e15
low = torch.tensor([min_M200c/1.0e15], device = device)
high = torch.tensor([max_M200c/1.0e15], device = device)
prior = utils.BoxUniform(low=low, high=high)
simulator, prior = prepare_for_sbi(simulator_func_train, prior)

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

#train the density estimator
density_estimator = inference.train(max_num_epochs=1000)

#build the posterior
posterior = inference.build_posterior(density_estimator)

def get_mean_std(x, lnPx):
    #Given a PDF, compute its mean and standard deviation
    dx = x[1:]-x[:-1]
    Px = np.exp(lnPx - np.max(lnPx))
    norm = np.sum(0.5*dx*(Px[1:] + Px[:-1]))
    Px = Px/norm
    mean_integrand = x*Px
    mean = np.sum(0.5*dx*(mean_integrand[1:] + mean_integrand[:-1]))
    std_integrand = (x-mean)**2*Px
    std = np.sqrt(np.sum(0.5*dx*(std_integrand[1:] + std_integrand[:-1])))
    return mean, std

#Useful for plots
def get_stacked_posteriors(N_trials, M200c_true):
    #Generate N_trials mock data sets corresponding to M200c_true
    #then analyze this using the likelihood and SBI
    #return the stacked likelihoods and the mean and std of the mass estimates
    num_M200c = 50 #number of M200c values to evaluate likelihood at
    M200c_arr = torch.linspace(min_M200c, max_M200c, num_M200c)
    true_lnlike_mat = np.zeros((N_trials, num_M200c))
    sbi_lnlike_mat = np.zeros((N_trials, num_M200c))
    for triali in range(0, N_trials):
        #Generate a mock data set with default mass
        x_0 = simulator_func_test(torch.tensor([M200c_true/1.0e15]).float())
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
        #Get mean and variance from stacked likelihoods
        dM = M200c_arr[1:]-M200c_arr[:-1]
        stacked_sbi_lnlike = np.sum(sbi_lnlike_mat, axis = 0)
        stacked_true_lnlike = np.sum(true_lnlike_mat, axis = 0)
        mean_sbi, std_sbi = get_mean_std(M200c_arr.numpy(), stacked_sbi_lnlike)
        mean_like, std_like = get_mean_std(M200c_arr.numpy(), stacked_true_lnlike)

    stacked_sbi_like = np.exp(stacked_sbi_lnlike - np.max(stacked_sbi_lnlike))
    stacked_true_like = np.exp(stacked_true_lnlike - np.max(stacked_true_lnlike))


    return M200c_arr, stacked_sbi_like, stacked_true_like, \
        mean_sbi, std_sbi, \
        mean_like, std_like
print("starting plot 1")
#Plot 1: stacked likelihood and plot for a single set of 10 clusters
N_clusters = 20
M200c_arr, stacked_sbi_like, stacked_true_like, _, _, _, _ = get_stacked_posteriors(N_clusters, M200c_default)
fig, ax = pl.subplots(1,1, figsize = (8,6))
ax.plot(M200c_arr, stacked_sbi_like, label = r'${\rm SBI}$', lw = 3, color = 'dodgerblue')
ax.plot(M200c_arr, stacked_true_like, label = r'${\rm Exact\,Likelihood}$', lw = 3, ls = 'dashed', color = 'orangered')
ax.plot([M200c_default, M200c_default], [0., 1.], label = r'${\rm True\,mass}$', color = 'black', lw = 3, ls = 'dotted')
ax.set_xlabel('M200c')
ax.set_ylabel('lnlike')
ax.legend(fontsize = 14)
fig.savefig('./figs/SBI_massonly_Npix{}_Nsims{}_method{}_train{}_test{}.png'.format(N_pix, num_sims, method, lensing_type_train, lensing_type_test))

'''
print("starting plot 2")
# Plot 2: Trials at different 'truth' masses.
# For each trial, we generate mock data and anayze using likelihood and SBI.
# Then we compute corresponding mean and std.
num_true_mass = 20
true_mass_arr = np.linspace(min_M200c, max_M200c, num_true_mass)
mean_M200c_sbi_arr = np.zeros(num_true_mass)
std_M200c_sbi_arr = np.zeros(num_true_mass)
mean_M200c_like_arr = np.zeros(num_true_mass)
std_M200c_like_arr = np.zeros(num_true_mass)
for ti in range(0,num_true_mass):
    _, _, _, mean_sbi, std_sbi, mean_like, std_like = get_stacked_posteriors(N_clusters, true_mass_arr[ti])
    mean_M200c_sbi_arr[ti] = mean_sbi
    std_M200c_sbi_arr[ti] = std_sbi
    mean_M200c_like_arr[ti] = mean_like
    std_M200c_like_arr[ti] = std_like

#Plot 2: mean and std of mass estimates as a function of true mass
fig, ax = pl.subplots(1,1, figsize = (8,6))
ax.errorbar(true_mass_arr, mean_M200c_sbi_arr, yerr = std_M200c_sbi_arr, label = r'${\rm SBI}$', lw = 3, color = 'dodgerblue', capsize = 3)
ax.errorbar(true_mass_arr+8.0e13, mean_M200c_like_arr, yerr = std_M200c_like_arr, label = r'${\rm Exact\,Likelihood}$', lw = 3, ls = 'dashed', color = 'orangered', capsize = 3)
ax.set_xlabel(r'${\rm True\,M200c}$')
ax.legend()
ax.plot([0., 0.], [1.0e16, 1.0e16], color= 'black')
fig.savefig('./figs/SBI_multi_massonly_Npix{}_Nsims{}_method{}.png'.format(N_pix, num_sims, method))
'''