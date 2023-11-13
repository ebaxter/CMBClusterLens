import map_funcs
import numpy as np
import pdb

import matplotlib.pyplot as pl

def apply_beam_and_transfer(map_in, map_settings, obs_settings, make_plots = False):
    """Apply beam and transfer functions to map."""
    map_fft = np.fft.fft2(map_in)
    lx, ly = map_funcs.get_lxly(map_settings)
    lsquare = lx**2. + ly**2.
    #prevent divide by zero error
    bad = np.where(lsquare == 0)
    lsquare[bad] = 1.0
    
    theta_beam = obs_settings['beam_fwhm_arcmin']/60.0*np.pi/180.0
    l_beam = 4.*np.sqrt(np.log(2.))/theta_beam
    beam = np.exp(-(lsquare/l_beam**2.))
    tf = np.ones(beam.shape)
    above_lxmax = np.where(lx > obs_settings['lx_max'])
    tf[above_lxmax] = 0.
    map_out = np.real(np.fft.ifft2(map_fft*beam*tf))

    if make_plots:  
        fig, ax = pl.subplots(1,3, figsize = (12,4))
        ax[0].imshow(map_in)
        ax[0].set_title('cmb map in')
        ax[1].imshow(beam)
        ax[1].set_title('beam')
        ax[2].imshow(np.real(map_out))
        ax[2].set_title('map after beam and tf')
        fig.savefig('./figs/beam_test.png')

    return map_out

def generate_noise_map(map_settings, obs_settings):
    #Generate white noise at desired level
    #noise variance in a pixel
    noise_var_muk2 = (obs_settings['noise_mukarcmin'] / map_settings['pix_size_arcmin'])**2
    N_pix = map_settings['N_pix']
    noise_map = np.random.normal(loc = 0.0, scale = np.sqrt(noise_var_muk2), size = (N_pix, N_pix))
    return noise_map