import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
from colossus.halo import mass_so


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

#Compute the deflection field using FFTs, and compare to analytic formula
def test_deflection(M200c, c200c, map_settings, cluster_lensing_settings, cosmo_params):
    Npix = map_settings['N_pix']
    hubble = cosmo_params['H0']/100.
    params = np.array([M200c, c200c])
    cluster_kappa = sim.generate_cluster_kappa(params, map_settings, cluster_lensing_settings, cosmo_params)
    Dx, Dy = lensing_funcs.get_deflection_from_kappa_singlemap(cluster_kappa, map_settings)
    theta_x, theta_y = map_funcs.get_theta_maps(map_settings)

    #Plot deflection field vs. analytic model
    fig, ax = pl.subplots(1,1)
    theta_x_1d = theta_x[int(Npix/2),:]
    ax.plot(theta_x_1d*60.*180/np.pi, Dx[int(Npix/2),:]*60.*180/np.pi, label = 'FFT calculation')
    ax.set_xlabel('angle from cluster center (arcmin)')
    ax.set_ylabel('deflection (arcmin)')
    #Calculate deflection field from analytic formula
    #See Eq. 6 in https://arxiv.org/pdf/astro-ph/0402314.pdf
    d_L = cluster_lensing_settings['dA_L']
    d_S = cluster_lensing_settings['dA_S']
    d_LS = cluster_lensing_settings['dA_LS']
    A_scaling = 0.25*M200c*u.Msun/(np.log(1.+c200c) - c200c/(1.+c200c)) #units of mass
    #note that there is an error in Dodelson formula - should be c^2 in denominator
    R200c = mass_so.M_to_R(M200c, cluster_lensing_settings['z_cluster'], '200c')*u.kpc
    prefactor = -16.*np.pi*const.G*A_scaling/((const.c**2)*R200c)*(d_LS/d_S)
    xx = (d_L*theta_x_1d*c200c/R200c).to('').value
    g_factor = g_of_x(np.abs(xx))
    Dx_analytic = prefactor.to('')*g_factor*np.sign(theta_x_1d) 
    ax.plot(theta_x_1d*60.*180/np.pi, Dx_analytic*60.*180/np.pi, label = 'analytic')
    ax.legend()

    fig.savefig('./figs/deflection_test.png')
    print("Defelection angle test plot saved")

if __name__ == '__main__':
    make_plots = True
    #Run all the slow calculations (e.g. cosmology stuff, power spectra)
    z_cluster = 0.5
    N_pix_kappa = 128
    N_pix_CMB = 16
    pix_size_arcmin_CMB = 0.5
    pix_size_arcmin_kappa = 0.5
    obs_type = 'spt3g_nobeam'
    all_settings = precompute.prepare_analysis(z_cluster, prep_likelihood = True, obs_type = obs_type, \
                                           N_pix_CMB = N_pix_CMB, N_pix_kappa = N_pix_kappa, \
                                            pix_size_arcmin_CMB = pix_size_arcmin_CMB,\
                                            pix_size_arcmin_kappa = pix_size_arcmin_kappa, make_plots = True)

    print("Precompute finished, Analysis ready to go")
    cluster_lensing_settings = all_settings['cluster_lensing_settings']
    map_settings_cmb = all_settings['cmb_map_settings']
    map_settings_kappa = all_settings['kappa_map_settings']
    obs_settings = all_settings['obs_settings']
    spectra = all_settings['spectra']
    cosmo_params = all_settings['cosmo_params']
    likelihood_info = all_settings['likelihood_info']

    #run test of deflection field
    print("running deflection test")
    hubble = cosmo_params['H0']/100.
    M200c = 3.0e14/hubble
    c200c = 5
    test_deflection(M200c, c200c, map_settings_kappa, cluster_lensing_settings, cosmo_params)

    #run test of map generation and lensing
    print("running lensing test")
    params = np.array([M200c, c200c])
    lensing_type = 'simple'
    generate_from_cov = True
    lensed_map, unlensed_map = sim.generate_lensed_map(params, cluster_lensing_settings, map_settings_cmb, map_settings_kappa, obs_settings, \
                                        spectra, cosmo_params, make_plots = False, return_unlensed = True, lensing_type = lensing_type, \
                                            generate_from_cov = generate_from_cov, likelihood_info = likelihood_info)
    
