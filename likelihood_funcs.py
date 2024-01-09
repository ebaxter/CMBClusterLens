import pdb
import numpy as np
import scipy.linalg as linalg

#For likelihood calculation
from scipy.special import jv

import map_funcs
import sim_cmb_cluster_lens as sim
import lensing_funcs

#Use Bessel function approx to get covariance element
def get_cov_element(ell, C_ell, ang_sep):
    W_ell = jv(0,ell*(ang_sep))
    cov_element = np.sum((1./(4.*np.pi))*(2.*ell + 1.)*C_ell*W_ell)
    return cov_element

#Exact likelihood. We use unlensed CMB power spectrum to calculate covariance elements
#We assume that lensing by LSS occurs at redshifts above the cluster
def lnlikelihood(params, cluster_settings, cmb_map_settings, kappa_map_settings, obs_settings, spectra, \
                        cosmo_params, likelihood_info, obs_data, print_output = False, use_pcs = False, use_unlensedcltt = False, restrict_map_pix = -1):
    if use_pcs:
        print("PC-based likelihood not implemented")
        #N_pcs = 5
        #return lnlikelihood_pcs(params, N_pcs, cluster_settings, map_settings, obs_settings, spectra, \
        #                cosmo_params, likelihood_info, obs_data, print_output = print_output)

    x_map, y_map = map_funcs.get_theta_maps(cmb_map_settings)
    N_pix = cmb_map_settings['N_pix']
    cluster_kappa = sim.generate_cluster_kappa(params, kappa_map_settings, cluster_settings, cosmo_params)
    Dx, Dy = lensing_funcs.get_deflection_from_kappa_multimap(cluster_kappa, kappa_map_settings, cmb_map_settings)

    #by default, we use the lensed CMB power spectrum to calculate covariance elements
    if use_unlensedcltt:
        cov_interp_func = likelihood_info['cov_interp_func_unlensed']
    else:
        cov_interp_func = likelihood_info['cov_interp_func_lensed']

    if (restrict_map_pix > 0):
        #restrict to central region of map
        min_index = int(N_pix/2. - restrict_map_pix/2.)
        max_index = int(N_pix/2. + restrict_map_pix/2.)
        x_map = x_map[min_index:max_index, min_index:max_index]
        y_map = y_map[min_index:max_index, min_index:max_index]
        
        Dx = Dx[min_index:max_index, min_index:max_index]
        Dy = Dy[min_index:max_index, min_index:max_index]

    angsep_mat = map_funcs.get_angsep_mat(x_map, y_map, Dx, Dy)
    #print("like M200 = ", params[0])
    #print("var Dx = ", np.var(Dx))
    cov_cmb = cov_interp_func(angsep_mat)

    noise_var_muk2 = (obs_settings['noise_mukarcmin'] / cmb_map_settings['pix_size_arcmin'])**2
    cov_noise = np.diag(noise_var_muk2 + np.zeros(cov_cmb.shape[0]))
    cov = cov_cmb + cov_noise
    #get log determinant of covariance matrix
    sign, lndetcov = np.linalg.slogdet(cov)
    
    if (np.isnan(lndetcov)):
        return -1.0e10
    else:
        invcov = linalg.inv(cov)
        term1 = -0.5*lndetcov

        term2 = -0.5*(np.dot(obs_data.flatten(),np.dot(invcov, obs_data.flatten())))
        lnlikelihood =  term1 + term2 #up to additive factors that don't depend on params

    return lnlikelihood, term1, term2


'''
def lnlikelihood_pcs(params, N_pc, cluster_settings, map_settings, obs_settings, spectra, \
                        cosmo_params, likelihood_info, obs_data, print_output = False):
    #compute likelihood by integrating over principal component amplitudes, using the first N_pc components
    x_map, y_map = map_funcs.get_theta_maps(map_settings)
    cluster_kappa = sim.generate_cluster_kappa(params, map_settings, cluster_settings, cosmo_params)
    Dx, Dy = lensing_funcs.get_deflection_from_kappa(cluster_kappa, map_settings)

    cov_interp_func = likelihood_info['cov_interp_func']
    angsep_mat = map_funcs.get_angsep_mat(x_map, y_map, Dx, Dy)
    #print("like M200 = ", params[0])
    #print("var Dx = ", np.var(Dx))
    cov_cmb = cov_interp_func(angsep_mat)

    #Determine principal components of unlensed CMB
    uu, vv = np.linalg.eig(cov_cmb)

    noise_var_muk2 = (obs_settings['noise_mukarcmin'] / map_settings['pix_size_arcmin'])**2

    grid_res = 100
    '''

