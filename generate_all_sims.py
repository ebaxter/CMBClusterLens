import sim_cmb_cluster_lens as sim
import precompute
import settings
import numpy as np
import pdb

if __name__ == '__main__':
    suffix = '1130'

    N_sims = 5000 #how many sims to generate
    
    param_scaling = np.array([1.0e15,1.0]) #scale all parameters by this amount (mass, concentration)

    settings = settings.load_settings()
    N_pix = settings['N_pix']
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
    param_min = np.array([1.0e13, 1.0])/param_scaling
    param_max = np.array([3.0e15, 10.0])/param_scaling

    #Run simulations across the prior space
    dim = len(param_min)
    all_param = np.zeros((N_sims, dim))
    N_pix = map_settings['N_pix']
    all_data = np.zeros((N_sims, N_pix, N_pix))
    all_data_unlensed = np.zeros((N_sims, N_pix, N_pix))

    output_filename = './sims/sims_lensingtype{}_scaled_generatefromcov{}_Nsims{}_Npix{}_{}.npz'.format(lensing_type, generate_from_cov,N_sims, N_pix, suffix)
    print("will save to ", output_filename)
    for pi in range(N_sims):
        print("pi= ", pi)
        params_scaled = np.random.uniform(param_min, param_max)
        params = params_scaled*param_scaling
        lensed_map, unlensed_map = sim.generate_lensed_map(params, cluster_settings, map_settings, obs_settings, \
                                     spectra, cosmo_params, make_plots = False, return_unlensed = True, lensing_type = lensing_type, \
                                        generate_from_cov = generate_from_cov, likelihood_info = likelihood_info)
        all_param[pi, :] = params_scaled
        all_data[pi, :, :] = lensed_map
        all_data_unlensed[pi,:, :] = unlensed_map

    np.savez_compressed(output_filename, params=all_param, data=all_data, \
                        data_unlensed = all_data_unlensed, \
                        param_min = param_min, param_max = param_max, param_scaling = param_scaling)
