import numpy as np
import pdb

import time
import precompute
import likelihood_funcs
import lensing_funcs
import map_funcs
import sim_cmb_cluster_lens as sim

import astropy.units as u
import astropy.constants as const

import matplotlib.pyplot as pl

#useful for computing deflection field due to NFW profile
def g_of_x(x):
    #Eq. 7 of https://arxiv.org/pdf/astro-ph/0402314.pdf
    ltone = np.where(x < 1)[0]
    gtone = np.where(x > 1)[0]
    g_x = np.zeros(x.shape)
    g_x[ltone] = (1./x[ltone])*(np.log(x[ltone]/2.)+np.log(x[ltone]/(1-np.sqrt(1-x[ltone]**2.)))/np.sqrt(1-x[ltone]**2.))
    g_x[gtone] = (1./x[gtone])*(np.log(x[gtone]/2.) + (np.pi/2. - np.arcsin(1./x[gtone]))/np.sqrt(x[gtone]**2. - 1.))
    return g_x

def test_deflection(map_settings, cluster_settings, cosmo_params):
    Npix = map_settings['N_pix']
    hubble = cosmo_params['H0']/100.
    M200c = 1.0e15/hubble
    c200c = 5
    params = np.array([M200c, c200c])
    cluster_kappa = sim.generate_cluster_kappa(params, map_settings, cluster_settings, cosmo_params)
    Dx, Dy = lensing_funcs.get_deflection_from_kappa(cluster_kappa, map_settings)
    theta_x, theta_y = map_funcs.get_theta_maps(map_settings)

    #Plot deflection field vs. analytic model
    fig, ax = pl.subplots(1,1)
    theta_x_1d = theta_x[int(Npix/2),:]
    ax.plot(theta_x_1d*60.*180/np.pi, Dx[int(Npix/2),:]*60.*180/np.pi, label = 'FFT calculation')
    ax.set_xlabel('angle from cluster center (arcmin)')
    ax.set_ylabel('deflection (arcmin)')
    #Calculate deflection field from analytic formula
    #See Eq. 6 in https://arxiv.org/pdf/astro-ph/0402314.pdf
    d_L = cluster_settings['dA_L']
    d_S = cluster_settings['dA_S']
    d_LS = cluster_settings['dA_LS']
    R200c = cluster_settings['R200c']
    A_scaling = 0.25*M200c*u.Msun/(np.log(1.+c200c) - c200c/(1.+c200c)) #units of mass
    #note that there is an error in Dodelson formula - should be c^2 in denominator
    prefactor = -16.*np.pi*const.G*A_scaling/((const.c**2)*R200c)*(d_LS/d_S)
    xx = (d_L*theta_x_1d*c200c/R200c).to('').value
    g_factor = g_of_x(np.abs(xx))
    Dx_analytic = prefactor.to('')*g_factor*np.sign(theta_x_1d) 
    ax.plot(theta_x_1d*60.*180/np.pi, Dx_analytic*60.*180/np.pi, label = 'analytic')
    ax.legend()

    print("Defelection angle test plot saved")
    fig.savefig('./figs/deflection_test.png')

if __name__ == '__main__':
    #fix random seed
    #np.random.seed(seed=112345)

    make_plots = True
    #Run all the slow calculations (e.g. cosmology stuff, power spectra)
    z_cluster = 0.5
    all_settings = precompute.prepare_analysis(z_cluster, prep_likelihood = True, obs_type = 'spt3g_nobeam', make_plots = make_plots)
    print("Analysis ready to go")
    cluster_settings = all_settings['cluster_settings']
    map_settings = all_settings['map_settings']
    obs_settings = all_settings['obs_settings']
    spectra = all_settings['spectra']
    cosmo_params = all_settings['cosmo_params']
    likelihood_info = all_settings['likelihood_info']

    #True parameters for mock data set
    M200c = 3.0e14
    c200c = 5
    true_params = np.array([M200c, c200c])
    t1 = time.time()

    #run test of deflection field
    #test_deflection(map_settings, cluster_settings, cosmo_params)

    #Generate a single lensed CMB map
    lensed_map = sim.generate_lensed_map(true_params, cluster_settings, map_settings, obs_settings, \
                                     spectra, cosmo_params, make_plots = make_plots, lensing_type = 'simple', likelihood_info = likelihood_info, generate_from_cov = True)
    t2 = time.time()
    print("full time to generate lensed cmb realization = ", t2-t1)

    #Do a single likelihood calculation at true parameters
    test_params = np.copy(true_params)
    lnlike, term1, term2 = likelihood_funcs.lnlikelihood(test_params, cluster_settings, map_settings, obs_settings, spectra, \
                        cosmo_params, likelihood_info, lensed_map, print_output = True)
    print("lnlike = ", lnlike)
    
    #Do a single likelihood calculation at incorrect parameters
    test_params = np.array([1.0e15, 5])
    lnlike_bad, term1_bad, term2_bad = likelihood_funcs.lnlikelihood(test_params, cluster_settings, map_settings, obs_settings, spectra, \
                        cosmo_params, likelihood_info, lensed_map, print_output = True)
    print("lnlike_bad = ", lnlike_bad)

    #Do a likelihood grid across multiple trials
    num_trials = 20
    min_M200c = 1e13
    max_M200c = 10.0e14
    num_M200c = 20
    M200c_arr = np.logspace(np.log10(min_M200c), np.log10(max_M200c), num_M200c)
    lnlike_mat = np.zeros((num_trials, num_M200c))
    for ti in range(0,num_trials):
        print("trial ", ti)
        lensed_map = sim.generate_lensed_map(true_params, cluster_settings, map_settings, obs_settings, \
                                     spectra, cosmo_params, make_plots = make_plots, lensing_type = 'simple', likelihood_info = likelihood_info, generate_from_cov = True)
        lnlike_arr = np.zeros(num_M200c)
        term1_arr = np.zeros(num_M200c)
        term2_arr = np.zeros(num_M200c)
        for mi in range(0,num_M200c):
            params = np.array([M200c_arr[mi], c200c])
            lnlike, term1, term2 = likelihood_funcs.lnlikelihood(params, cluster_settings, map_settings, obs_settings, spectra, \
                            cosmo_params, likelihood_info, lensed_map)
            lnlike_arr[mi] = lnlike
            term1_arr[mi] = term1
            term2_arr[mi] = term2
        lnlike_mat[ti,:] = lnlike_arr

    #plot the likelihood as a function of M200c
    '''
    fig, ax = pl.subplots(4,1)
    for ti in range(0,num_trials):
        ax[0].plot(M200c_arr, lnlike_mat[ti,:])
        ax[0].set_xscale('log')
    #ax[0].plot(M200c_arr, lnlike_arr)
    ax[0].set_ylabel('lnlike')
    ax[1].plot(M200c_arr, term1_arr)
    ax[1].set_ylabel('term1')
    ax[1].set_xscale('log')
    ax[2].plot(M200c_arr, term2_arr)
    ax[2].set_ylabel('term2')
    ax[2].set_xscale('log')
    stacked_lnlikelihood = np.sum(lnlike_mat, axis = 0)
    ax[3].plot(M200c_arr, np.exp(stacked_lnlikelihood - np.max(stacked_lnlikelihood)))
    ax[3].plot([M200c, M200c], [0, 1.], label = 'true M200c')
    #ax[3].set_xscale('log')
    ax[3].set_ylabel('stacked lnlike')
    fig.tight_layout()
    fig.savefig('./figs/likelihood_grid.png')
    '''
    fig, ax = pl.subplots(1,1)
    stacked_lnlikelihood = np.sum(lnlike_mat, axis = 0)
    ax.plot(M200c_arr, np.exp(stacked_lnlikelihood - np.max(stacked_lnlikelihood)))
    ax.plot([M200c, M200c], [0, 1.], label = 'true M200c')
    ax.set_ylabel('stacked like')
    fig.tight_layout()
    fig.savefig('./figs/likelihood_grid.png')
