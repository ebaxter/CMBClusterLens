import map_funcs
import numpy as np
import pdb
import time
from scipy import interpolate as interp

def get_deflection_from_kappa(kappa_map, map_settings):
    #Warning: should we be applying some apodization here?

    #Get grids of lx and ly
    lx, ly = map_funcs.get_lxly(map_settings)
    lsquare = lx**2. + ly**2.

    #prevent divide by zero error
    bad = np.where(lsquare == 0)
    lsquare[bad] = 1.0

    N_pix = map_settings['N_pix']
    pix_size = map_settings['pix_size_arcmin']/60.0*np.pi/180.0
    #lensing potential
    # Warning: should this be L(L+1) instead of lsquare?
    pfft = np.fft.fft2(kappa_map)*2./lsquare    
    #deflection = grad phi
    gpx    = np.fft.ifft2( pfft * lx * -1.j)
    gpy    = np.fft.ifft2( pfft * ly * -1.j)
    return -np.real(gpx), -np.real(gpy)

def lens_map(map_in, kappa_map, map_settings, make_plots = False):
    """Lenses a map by a lensing field."""
    Npix = map_settings['N_pix']
    #Get deflection field maps
    ta = time.time()
    Dx, Dy = get_deflection_from_kappa(kappa_map, map_settings)
    x_map, y_map = map_funcs.get_theta_maps(map_settings)
    x_side, y_side = x_map[0,:], y_map[:,0]
    tb = time.time()
    unlensed_spline = interp.RectBivariateSpline(x_side, y_side, map_in, kx = 5, ky = 5)
    tc = time.time()

    #evaluate interpolation function at undeflected pixel positions
    newx = (x_map + Dx).flatten()
    newy = (y_map + Dy).flatten()
    lensed = unlensed_spline.ev(newx, newy).reshape(Npix, Npix).transpose()

    #plot the deflection fields and more
    if make_plots:
        lx, ly = map_funcs.get_lxly(map_settings)
        fig, ax = pl.subplots(1,8, figsize = (16,4))
        ax[0].imshow(x_map)
        ax[0].set_title('x map')
        ax[1].imshow(lx)
        ax[1].set_title('Lx')
        ax[2].imshow(map_in)
        ax[2].set_title('cmb map in')
        ax[3].imshow(kappa_map)
        ax[3].set_title('kappa map')
        ax[4].imshow(Dx)
        ax[4].set_title('Dx')
        ax[5].imshow(Dy)
        ax[5].set_title('Dy')
        ax[6].imshow(lensed)
        ax[6].set_title('lensed map')
        ax[7].imshow(lensed - map_in)
        ax[7].set_title('lensed - unlensed')
    
        fig.savefig('./figs/lensing_test.png')
        print("lensing test plot saved")

    return lensed
