import numpy as np

def get_theta_maps(map_settings):
    """Get theta maps in radians."""
    N_pix = map_settings['N_pix']
    pix_size_arcmin = map_settings['pix_size_arcmin']
    x_row = (np.arange(N_pix) - 0.5*(N_pix-1.))*pix_size_arcmin +  map_settings['map_center_x_arcmin']
    y_row = (np.arange(N_pix) - 0.5*(N_pix-1.))*pix_size_arcmin +  map_settings['map_center_y_arcmin']
    x_map = np.outer(np.ones(N_pix), x_row)
    y_map = np.outer(y_row, np.ones(N_pix))
    #Convert to radians
    return x_map*np.pi/180./60., y_map*np.pi/180./60.

#WARNING: this is extremely slow and memory intensive for large maps
def get_angsep_mat(theta_x_mat, theta_y_mat, dx_mat, dy_mat):
    N_pix = len(theta_x_mat)
    #Get angular separation between all pairs of pixels as in the absence of lensing deflection
    theta_x_mat_undolens = theta_x_mat + dx_mat
    theta_y_mat_undolens = theta_y_mat + dy_mat
    x1 = np.tile(theta_x_mat_undolens.flatten(),(N_pix**2,1))
    y1 = np.tile(theta_y_mat_undolens.flatten(),(N_pix**2,1))
    angsep_mat = np.sqrt((x1 - x1.transpose())**2. + (y1-y1.transpose())**2.)
    return angsep_mat

def get_lxly(map_settings):
    N_pix = map_settings['N_pix']
    dx = map_settings['pix_size_arcmin']/60.0*np.pi/180.0
    lx, ly = np.meshgrid( np.fft.fftfreq( N_pix, dx )*2.*np.pi,
                        np.fft.fftfreq( N_pix, dx )*2.*np.pi)
    return lx, ly

if __name__ == '__main__':
    print("runing map funcs")