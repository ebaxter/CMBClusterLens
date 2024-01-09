import map_funcs
import numpy as np
import pdb
import time
from scipy import interpolate as interp

def get_deflection_from_kappa_singlemap(kappa_map, map_settings):
    #Get grids of lx and ly
    lx, ly = map_funcs.get_lxly(map_settings)
    lsquare = lx**2. + ly**2.

    #prevent divide by zero error
    bad = np.where(lsquare == 0)
    lsquare[bad] = 1.0

    N_pix = map_settings['N_pix']
    pix_size = map_settings['pix_size_arcmin']/60.0*np.pi/180.0
    # kappa = (1/2) grad^2 phi
    # Warning: should this be L(L+1) instead of lsquare?
    pfft = 2.*np.fft.fft2(kappa_map)/lsquare    
    #deflection = grad phi
    gpx    = np.fft.ifft2( pfft * lx * -1.j)
    gpy    = np.fft.ifft2( pfft * ly * -1.j)
    return -np.real(gpx), -np.real(gpy)

def get_deflection_from_kappa_multimap(kappa_map, kappa_map_settings, cmb_map_settings):
    # Allow for different kappa and CMB grids.  Output deflection field 
    # will have same dimensions as CMB map

    #coordinates in kappa map    
    kappa_x_map, kappa_y_map = map_funcs.get_theta_maps(kappa_map_settings)
    kappa_x_side, kappa_y_side = kappa_x_map[0,:], kappa_y_map[:,0]
    N_pix_kappa = kappa_map_settings['N_pix']

    #coordinates in CMB map
    x_map, y_map = map_funcs.get_theta_maps(cmb_map_settings)
    x_side, y_side = x_map[0,:], y_map[:,0]
    N_pix_cmb = cmb_map_settings['N_pix']

    #Get deflection field maps on kappa map grid
    Dx_kappamap, Dy_kappamap = get_deflection_from_kappa_singlemap(kappa_map, kappa_map_settings)
        
    #if pixel sizes aren't the same for CMB and kappa maps, we need to interpolate
    if (kappa_map_settings['pix_size_arcmin'] != cmb_map_settings['pix_size_arcmin']):
        #interpolator for deflection field
        Dx_spline = interp.RectBivariateSpline(kappa_x_side, kappa_y_side, Dx_kappamap, kx = 5, ky = 5)
        Dy_spline = interp.RectBivariateSpline(kappa_x_side, kappa_y_side, Dy_kappamap, kx = 5, ky = 5)
        #Evaluate deflection field on CMB map grid
        Dx = Dx_spline.ev(x_map, y_map).reshape(N_pix_cmb, N_pix_cmb).transpose()
        Dy = Dy_spline.ev(x_map, y_map).reshape(N_pix_cmb, N_pix_cmb).transpose()
    else:
        #if pixel sizes are the same, we can directly extract the deflection field on the desired map
        min_index_kappa_map = np.where(kappa_x_side == x_side[0])[0][0]
        max_index_kappa_map = np.where(kappa_x_side == x_side[-1])[0][0] + 1
        Dx = Dx_kappamap[min_index_kappa_map:max_index_kappa_map, min_index_kappa_map:max_index_kappa_map]
        Dy = Dy_kappamap[min_index_kappa_map:max_index_kappa_map, min_index_kappa_map:max_index_kappa_map]
        
    return Dx, Dy

def lens_map(map_in, kappa_map, cmb_map_settings, kappa_map_settings, make_plots = False):
    """Lenses a map by a lensing field.  Kappa map can be defined on different grid than CMB map"""
    Npix = cmb_map_settings['N_pix']

    #coordinates in kappa map    
    kappa_x_map, kappa_y_map = map_funcs.get_theta_maps(kappa_map_settings)
    kappa_x_side, kappa_y_side = kappa_x_map[0,:], kappa_y_map[:,0]

    #coordinates in CMB map
    x_map, y_map = map_funcs.get_theta_maps(cmb_map_settings)
    x_side, y_side = x_map[0,:], y_map[:,0]    

    #Get deflection fields on CMB map grid
    Dx, Dy = get_deflection_from_kappa_multimap(kappa_map, kappa_map_settings, cmb_map_settings)

    #Set up interpolator for unlensed CMB map
    unlensed_spline = interp.RectBivariateSpline(x_side, y_side, map_in, kx = 5, ky = 5)    

    #evaluate interpolation function at undeflected pixel positions
    newx = (x_map + Dx).flatten()
    newy = (y_map + Dy).flatten()
    lensed = unlensed_spline.ev(newx, newy).reshape(Npix, Npix).transpose()

    return lensed
