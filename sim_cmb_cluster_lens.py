
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
from colossus.halo import profile_composite
from colossus.halo import concentration
from colossus.lss import bias



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
    #Set up coordinate maps
    thetax_map, thetay_map = map_funcs.get_theta_maps(map_settings)
    theta_map = np.sqrt(thetax_map**2. + thetay_map**2.)
    #Colossus uses kpc/h units for distances and Msun/h for mass
    #Physical separation from cluster center
    r_map = theta_map*cluster_settings['dA_L'].to('Mpc').value
    tiny = np.where(r_map < 1E-10)[0]
    r_map[tiny] = 1E-10

    #If we've precomputed cluster kappa as function of M and r, then use that
    if 'cluster_kappa_interp_func' in cluster_settings:
        kappa_map = cluster_settings['cluster_kappa_interp_func'](M200c, r_map)
        #otherwise, do full calculation
    else:
        hubble = cosmo_params['H0']/100.
        M200c = params[0]
        c200c = params[1]
        #handle case where M200c = 0
        if M200c == 0.:
            return np.zeros((map_settings['N_pix'], map_settings['N_pix']))

        #If c200c < 0, then use concentration-mass relation to fix concentration
        if c200c < 0.:
            hubble = cosmo_params['H0']/100.
            c200c = concentration.concentration(M200c*hubble, '200c', cluster_settings['z_cluster'], model = 'diemer19')

        #Set up Colossus profile object
        pure_NFW = True
        if pure_NFW:
            p = profile_nfw.NFWProfile(M = M200c*hubble, mdef = '200c', \
                                    z = cluster_settings['z_cluster'], c = c200c)
        else:
            #NFW + outer profile
            halo_bias = bias.haloBias(M200c*hubble, model = 'tinker10', z = cluster_settings['z_cluster'], mdef = '200c')
            p = profile_composite.compositeProfile('nfw', outer_names = ['mean', 'cf'],
                M = M200c*hubble, mdef = '200c', z = cluster_settings['z_cluster'], c = c200c, bias = halo_bias)             
        #Use Colossus to get surface density 
        max_r_interpolate = 20.*1000.*hubble #only integrate out to 20 Mpc when computing projected density
        Sigma_arr = u.Msun*hubble*p.surfaceDensity(r_map.flatten()*hubble*1000, max_r_interpolate = max_r_interpolate)/u.kpc**2
        Sigma_map = Sigma_arr.reshape(map_settings['N_pix'], map_settings['N_pix'])
        #Convert to kappa
        kappa_map = (Sigma_map/cluster_settings['Sigma_crit']).to('')
        return kappa_map

def generate_lensed_map(params, cluster_settings, cmb_map_settings, kappa_map_settings, obs_settings, spectra, \
                        cosmo_params, make_plots = False, return_unlensed = False, lensing_type = 'full', \
                            likelihood_info = None, generate_from_cov = False, fix_seed = False):
    #lensing convergence from cluster
    cluster_kappa = generate_cluster_kappa(params, kappa_map_settings, cluster_settings, cosmo_params)
    #If we generate from cov, we need angular separation matrices
    if generate_from_cov:
        cmb_x_map, cmb_y_map = map_funcs.get_theta_maps(cmb_map_settings)        
        cmb_angsep_mat = map_funcs.get_angsep_mat(cmb_x_map, cmb_y_map, 0., 0.)

    if (lensing_type == 'full'):
    # Full calculation separates high z and low z contributions from cluster.  
    # We generate a truly unlensed map to start
        if generate_from_cov:
            cov_interp_func = likelihood_info['cov_interp_func_unlensed']
            cov = cov_interp_func(cmb_angsep_mat)    
            N_pix_cmb = cmb_map_settings['N_pix']
            map_unl = np.random.multivariate_normal(np.zeros(N_pix_cmb**2), cov).reshape(N_pix_cmb, N_pix_cmb)
        if not generate_from_cov:
            #generate the unlensed map from fourier transforming white noise
            map_unl = generate_unlensed_map(spectra['ell'], spectra['clTT_unlensed'], cmb_map_settings)
        #Generate kappa maps for different sources of lensing
        lss_highz_kappa = generate_lss_kappa(spectra['bigell'], spectra['clpp_highz'], kappa_map_settings) 
        lss_lowz_kappa = generate_lss_kappa(spectra['bigell'],  spectra['clpp_lowz'], kappa_map_settings)
        # Lens the unlensed map by LSS at higher z than cluster
        map_at_cluster = lensing_funcs.lens_map(map_unl, lss_highz_kappa, cmb_map_settings, kappa_map_settings) 
        #lens the map by the cluster
        map_after_cluster = lensing_funcs.lens_map(map_at_cluster, cluster_kappa, cmb_map_settings, kappa_map_settings, make_plots = make_plots)
        #lens the map by LSS at lower z than cluster
        map_at_telescope = lensing_funcs.lens_map(map_after_cluster, lss_lowz_kappa, cmb_map_settings, kappa_map_settings)
    if (lensing_type == 'simple'):
        #simple calculation treats all lensing by LSS as occuring at high redshift above cluster
        if generate_from_cov:
            #In simple model, we use spectrum of LSS-lensed CMB
            cov_interp_func = likelihood_info['cov_interp_func_lensed']
            cov = cov_interp_func(cmb_angsep_mat)
            map_unl = np.random.multivariate_normal(np.zeros(cmb_map_settings['N_pix']**2), cov).reshape(cmb_map_settings['N_pix'], cmb_map_settings['N_pix'])
        if not generate_from_cov:
            map_unl = generate_unlensed_map(spectra['ell'], spectra['clTT_lensed'], map_settings)
        map_at_telescope = lensing_funcs.lens_map(map_unl, cluster_kappa, cmb_map_settings, kappa_map_settings)
        map_after_cluster = np.copy(map_at_telescope)        
        # Apply beam and transfer functions
    map_obs = obs_funcs.apply_beam_and_transfer(map_at_telescope, cmb_map_settings, obs_settings, make_plots = make_plots)
    # Add instrumental noise
    noise = obs_funcs.generate_noise_map(cmb_map_settings, obs_settings)
    map_out = map_obs + noise

    if return_unlensed:
        return map_out, map_unl
    else:
        return map_out

if __name__ == '__main__':
    print("functions to generate a cluster lensed CMB map")
    