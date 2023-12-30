import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import precompute
import likelihood_funcs
import lensing_funcs
import map_funcs
import sim_cmb_cluster_lens as sim
import settings

import pickle as pk
import numpy as np
import pdb


'''
Generate mock maps, then analyze by computing likelihood across grid
'''

if __name__ == '__main__':
    N_likelihood_calculations = 400 #how many mock data sets do we want to analyze
    suffix = '1205'
    make_plots = False
    use_pcs = False
    settings = settings.load_settings()
    print("settings: ", settings)
    N_pix = settings['N_pix']
    pix_size_arcmin = settings['pix_size_arcmin']
    generate_from_cov = settings['generate_from_cov']
    lensing_type = settings['lensing_type']
    obs_type = settings['obs_type']
    z_cluster = settings['z_cluster']

    #True parameters for mock data sets
    M200c = 3.0e14
    c200c = 5
    true_params = np.array([M200c, c200c])

    #Run all the slow calculations (e.g. cosmology stuff, power spectra)
    all_settings = precompute.prepare_analysis(z_cluster, prep_likelihood = True, \
                                obs_type = obs_type, make_plots = make_plots,  N_pix = N_pix, \
                                pix_size_arcmin = pix_size_arcmin)
    print("Analysis ready to go")
    cluster_settings = all_settings['cluster_settings']
    map_settings = all_settings['map_settings']
    obs_settings = all_settings['obs_settings']
    spectra = all_settings['spectra']
    cosmo_params = all_settings['cosmo_params']
    likelihood_info = all_settings['likelihood_info']
    N_pix = map_settings['N_pix']

    #Do a likelihood grid across multiple trials
    min_M200c = 1e13
    max_M200c = 10.0e14
    num_M200c = 15
    M200c_arr = np.logspace(np.log10(min_M200c), np.log10(max_M200c), num_M200c)

    min_c200c = 1.0
    max_c200c = 10.0
    num_c200c = 10
    c200c_arr = np.logspace(np.log10(min_c200c), np.log10(max_c200c), num_c200c)

    obs_map_list = []
    unlensed_map_list = []
    lnlike_mat_list = []

    #Loop over likelihood calculations
    for ti in range(0,N_likelihood_calculations):
        lnlike_mat = np.zeros((num_M200c, num_c200c))
        print("trial ", ti)
        lensed_map, unlensed_map = sim.generate_lensed_map(true_params, cluster_settings, map_settings, obs_settings, \
                                spectra, cosmo_params, make_plots = False, lensing_type = lensing_type, likelihood_info = likelihood_info, \
                                generate_from_cov = generate_from_cov, return_unlensed = True)

        #Compute likelihood on grid
        for mi in range(0,num_M200c):
            for ci in range(0,num_c200c):
                params = np.array([M200c_arr[mi], c200c_arr[ci]])
                lnlike, term1, term2 = likelihood_funcs.lnlikelihood(params, cluster_settings, map_settings, obs_settings, spectra, \
                            cosmo_params, likelihood_info, lensed_map, use_pcs = use_pcs)
                lnlike_mat[mi,ci] = lnlike
        obs_map_list.append(lensed_map)
        unlensed_map_list.append(unlensed_map)
        lnlike_mat_list.append(lnlike_mat)

    save_data = {'obs_map_list': obs_map_list, 'unlensed_map_list':unlensed_map_list, \
                 'lnlike_mat_list': lnlike_mat_list, 'M200c_arr': M200c_arr, 'c200c_arr': c200c_arr, \
                 'true_params': true_params, 'z_cluster': z_cluster, 'generate_from_cov': generate_from_cov, 'obs_type': obs_type}
    output_filename = './likelihood_grids/' + 'likelihood_grid_Npix{}_generatefromcov{}_usepcs{}_{}_num{}.pk'.format(N_pix,generate_from_cov, use_pcs, suffix, N_likelihood_calculations)
    pk.dump(save_data, open(output_filename, 'wb'))