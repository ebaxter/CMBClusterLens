
import torch
import numpy as np
import matplotlib.pyplot as pl
from scipy import interpolate as interp
import pdb
import time
import precompute
import map_funcs 
import lensing_funcs
import obs_funcs

from astropy import constants as const
from astropy import units as u


from colossus.halo import profile_nfw

#da are angular diameter distances
#chi are comoving distances
#settings are fixed throughout
#params can vary

# Map generation functions
def generate_map_from_Cells(ell, C_ell, map_settings):
    lx, ly = map_funcs.get_lxly(map_settings)
    lsquare = lx**2. + ly**2.
    in_lrange = np.where((ell < map_settings['lmax']) & (ell > map_settings['lmin']))[0]
    #Cl evaluated on 2d ell array
    CL2d = interp.interp1d(ell[in_lrange], C_ell[in_lrange], kind = 'linear', bounds_error = False, fill_value = 0.0)(np.sqrt(lsquare))

    #FFT of map with flat noise power spectrum
    whitenoise = np.fft.fft2(np.random.normal(0,1,(map_settings['N_pix'],map_settings['N_pix'])))    
    #Multiply by sqrt of power spectrum
    map_realization = np.fft.ifft2(whitenoise*np.sqrt(CL2d))
    #Cl has units of muK**2, map realization has units of muK
    map_realization = np.real(map_realization)

    return map_realization

def generate_unlensed_map(ell, C_ell, map_settings):
    """Generate an unlensed CMB map."""
    map_cmb = generate_map_from_Cells(ell, C_ell, map_settings)
    return map_cmb

def generate_lss_kappa(ell, clkk, map_settings):
    #do we need to do this at higher resolution than the CMB map?
    """Generate a Gaussian realization of clkk"""
    lensing_map_lss = generate_map_from_Cells(ell, clkk, map_settings)
    return lensing_map_lss
    
def generate_cluster_kappa(params, map_settings, cluster_settings, cosmo_params):
    hubble = cosmo_params['H0']/100.
    M200c = params[0]
    c200c = params[1]

    if M200c == 0.:
        return np.zeros((map_settings['N_pix'], map_settings['N_pix']))
    
    #Colossus calculation of lensing profile
    p = profile_nfw.NFWProfile(M = M200c*hubble, mdef = '200c', \
                               z = cluster_settings['z_cluster'], c = c200c)

    #This could be sped up by building interpolation function for lensing profile
    #Set up coordinate maps
    thetax_map, thetay_map = map_funcs.get_theta_maps(map_settings)
    theta_map = np.sqrt(thetax_map**2. + thetay_map**2.)
    #Physical separation from cluster center
    r_map = theta_map*cluster_settings['dA_L'].to('Mpc').value
    tiny = np.where(r_map < 1E-10)[0]
    r_map[tiny] = 1E-10
    #Colossus uses kpc/h units for distances and Msun/h for mass
    #7777 check factors of h
    #Use Colossus to get surface density
    Sigma_map = u.Msun*hubble*p.surfaceDensity(r_map*hubble*1000.)/u.kpc**2
    kappa_map = (Sigma_map/cluster_settings['Sigma_crit']).to('')
    return kappa_map

def generate_lensed_map(params, cluster_settings, map_settings, obs_settings, spectra, \
                        cosmo_params, make_plots = False, return_unlensed = False, lensing_type = 'full', likelihood_info = None, generate_from_cov = False, fix_seed = False):
    if generate_from_cov:
        #generate the unlensed map from covariance matrix instead of fourier transforming noise map
        x_map, y_map = map_funcs.get_theta_maps(map_settings)        
        cov_interp_func = likelihood_info['cov_interp_func']
        angsep_mat = map_funcs.get_angsep_mat(x_map, y_map, 0., 0.)
        cov = cov_interp_func(angsep_mat)    
        map_unl = np.random.multivariate_normal(np.zeros(map_settings['N_pix']**2), cov).reshape(map_settings['N_pix'], map_settings['N_pix'])

    if not generate_from_cov:
        #generate the unlensed map from fourier transforming white noise
        map_unl = generate_unlensed_map(spectra['ell'], spectra['clTT_unlensed'], map_settings)
    
    #full calculation separates high z and low z contributions from cluster
    if (lensing_type == 'full'):
        #Generate kappa maps for different sources of lensing
        t1 = time.time()
        cluster_kappa = generate_cluster_kappa(params, map_settings, cluster_settings, cosmo_params)
        lss_highz_kappa = generate_lss_kappa(spectra['bigell'], spectra['clpp_highz'], map_settings) 
        lss_lowz_kappa = generate_lss_kappa(spectra['bigell'],  spectra['clpp_lowz'], map_settings)
        t2 = time.time()
        # Lens the unlensed map by LSS at higher z than cluster
        map_at_cluster = lensing_funcs.lens_map(map_unl, lss_highz_kappa, map_settings) 
        t3 = time.time()
        map_after_cluster = lensing_funcs.lens_map(map_at_cluster, cluster_kappa, map_settings, make_plots = make_plots)
        map_at_telescope = lensing_funcs.lens_map(map_after_cluster, lss_lowz_kappa, map_settings)
        t4 = time.time()
    if (lensing_type == 'simple'):
        #simple calculation ignores LSS and considers only lensing by cluster        
        cluster_kappa = generate_cluster_kappa(params, map_settings, cluster_settings, cosmo_params)
        map_at_telescope = lensing_funcs.lens_map(map_unl, cluster_kappa, map_settings)
        map_after_cluster = np.copy(map_at_telescope)
        
    # Apply beam and transfer functions
    #map_obs = apply_beam_and_transfer(map_at_telescope, map_settings, obs_settings, make_plots = make_plots)
    map_obs = obs_funcs.apply_beam_and_transfer(map_at_telescope, map_settings, obs_settings, make_plots = make_plots)
    # Add instrumental noise
    noise = obs_funcs.generate_noise_map(map_settings, obs_settings)
    map_out = map_obs + noise
    if (make_plots):
        #lensing field maps
        fig, ax = pl.subplots(1, 3, figsize = (8,4))

        ax[0].imshow(cluster_kappa)
        ax[0].set_title('cluster convergence')

        #ax[1].imshow(lss_highz_kappa)
        #ax[1].set_title('highz convergence')

        #ax[2].imshow(lss_lowz_kappa)
        #ax[2].set_title('lowz convergence')

        fig.savefig('./figs/kappa_maps.png')

        #CMB maps
        fig, ax = pl.subplots(2,2)
        ax[0,0].imshow(map_unl)
        ax[0,0].set_title('Unlensed')
        ax[0,1].imshow(map_after_cluster)
        ax[0,1].set_title('After cluster')
        ax[1,0].imshow(map_obs)
        ax[1,0].set_title('after beam')
        #ax[1,1].imshow(noise)
        ax[1,1].imshow(map_after_cluster - map_unl)
        ax[1,1].set_title('after - before')

        fig.savefig('./figs/lensed_map.png')

    if return_unlensed:
        return map_out, map_unl
    else:
        return map_out

if __name__ == '__main__':
    do_likelihood = False
    #Run all the slow calculations (e.g. cosmology stuff, power spectra)
    z_cluster = 0.5
    all_settings = precompute.prepare_analysis(z_cluster, prep_likelihood = do_likelihood)
    cluster_settings = all_settings['cluster_settings']
    map_settings = all_settings['map_settings']
    obs_settings = all_settings['obs_settings']
    spectra = all_settings['spectra']
    cosmo_params = all_settings['cosmo_params']
    if (do_likelihood):
        likelihood_info = all_settings['likelihood_info']

    M200c = 1.0e13
    c200c = 5
    params = np.array([M200c, c200c])
    t1 = time.time()

    #Generate a single lensed CMB map
    lensed_map = generate_lensed_map(params, cluster_settings, map_settings, obs_settings, \
                                     spectra, cosmo_params, make_plots = False, lensing_type = 'simple')
    t2 = time.time()
    print("full time to generate lensed cmb realization = ", t2-t1)

    