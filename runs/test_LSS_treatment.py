import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as pl

import numpy as np
import pdb
#cluster lensing functions
import map_funcs
import lensing_funcs
import settings as settings_mod
import precompute
import sim_cmb_cluster_lens as sim
import likelihood_funcs
import obs_funcs

N_pix =  16
z_cluster = 0.5

settings = settings_mod.load_settings()
pix_size_arcmin = settings['pix_size_arcmin']
obs_type = settings['obs_type']

all_settings = precompute.prepare_analysis(z_cluster, prep_likelihood = True, obs_type = obs_type, N_pix = N_pix, pix_size_arcmin = pix_size_arcmin)
print("Analysis ready to go")

spectra = all_settings['spectra']
cosmo_params = all_settings['cosmo_params']

cluster_settings = all_settings['cluster_settings']
map_settings = all_settings['map_settings']
obs_settings = all_settings['obs_settings']
likelihood_info = all_settings['likelihood_info']

#'truth' values
M200c_true = 3.0e15
c200c_true = -1# Setting this to be negative ==> concentration computed from mass using M-c relation
params = np.array([M200c_true, c200c_true])
settings['N_pix'] = N_pix
pix_size_arcmin = settings['pix_size_arcmin']
#Generate unlensed CMB map
generate_from_cov = True
if generate_from_cov:
    x_map, y_map = map_funcs.get_theta_maps(map_settings)        
    angsep_mat = map_funcs.get_angsep_mat(x_map, y_map, 0., 0.)    
    cov_interp_func = likelihood_info['cov_interp_func_unlensed']
    cov = cov_interp_func(angsep_mat)    
    map_unl = np.random.multivariate_normal(np.zeros(map_settings['N_pix']**2), cov).reshape(map_settings['N_pix'], map_settings['N_pix'])
if not generate_from_cov:
    #generate the unlensed map by fourier transforming white noise
    map_unl = sim.generate_unlensed_map(spectra['ell'], spectra['clTT_unlensed'], map_settings)
#Generate kappa maps for different sources of lensing
cluster_kappa = sim.generate_cluster_kappa(params, map_settings, cluster_settings, cosmo_params)
lss_highz_kappa = sim.generate_lss_kappa(spectra['bigell'], spectra['clpp_highz'], map_settings) 
lss_lowz_kappa = sim.generate_lss_kappa(spectra['bigell'],  spectra['clpp_lowz'], map_settings)
# Lens the unlensed map by LSS at higher z than cluster
map_at_cluster = lensing_funcs.lens_map(map_unl, lss_highz_kappa, map_settings) 
map_after_cluster = lensing_funcs.lens_map(map_at_cluster, cluster_kappa, map_settings, make_plots = False)
map_at_telescope = lensing_funcs.lens_map(map_after_cluster, lss_lowz_kappa, map_settings)

#alternatively, lens the unlensed maps by all lss, then by the cluster
lss_kappa = lss_highz_kappa + lss_lowz_kappa
map_at_cluster_alt = lensing_funcs.lens_map(map_unl, lss_kappa, map_settings)
map_at_telescope_alt = lensing_funcs.lens_map(map_at_cluster_alt, cluster_kappa, map_settings)

# Add instrumental noise
noise = obs_funcs.generate_noise_map(map_settings, obs_settings)
map_obs = map_at_telescope + noise
map_obs_alt = map_at_telescope_alt + noise

fig, ax = pl.subplots(1, 3, figsize = (12,4))
ax[0].imshow(map_at_telescope, vmin = -200.0, vmax = 20.0)
ax[0].set_title('lensed map')
ax[1].imshow(map_at_telescope_alt, vmin = -200.0, vmax = 20.0)
ax[1].set_title('lensed map (alt)')
ax[2].imshow(map_at_telescope - map_at_telescope_alt, vmin = -20.0, vmax = 20.0)
ax[2].set_title('difference')
fig.savefig('figs/test_lss_treatment.png')

#compute the likelihood for the two maps as a function of M200c
num_M200c = 20
lnlike_arr = np.zeros(num_M200c)
lnlike_arr_alt = np.zeros(num_M200c)
M200c_arr = np.linspace(1.0e14, 1.0e16, num_M200c)
for i in range(num_M200c):
    params = np.array([M200c_arr[i], c200c_true])
    use_pcs = False
    lnlike, term1, term2 = likelihood_funcs.lnlikelihood(params, cluster_settings, map_settings, obs_settings, spectra, \
                cosmo_params, likelihood_info, map_obs, use_pcs = use_pcs)
    lnlike_alt, term1, term2 = likelihood_funcs.lnlikelihood(params, cluster_settings, map_settings, obs_settings, spectra, \
                cosmo_params, likelihood_info, map_obs_alt, use_pcs = use_pcs)   
    lnlike_arr[i] = lnlike
    lnlike_arr_alt[i] = lnlike_alt 
fig, ax = pl.subplots(1, 1, figsize = (8,4))
ax.plot(M200c_arr, np.exp(lnlike_arr - np.max(lnlike_arr)), label ='baseline')
ax.plot(M200c_arr, np.exp(lnlike_arr_alt - np.max(lnlike_arr_alt)), label = 'alt')
ax.legend()
fig.savefig('figs/test_lss_treatment_lnlike.png')
pdb.set_trace()