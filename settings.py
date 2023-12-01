def load_settings(settings_type = 'default'):
    if settings_type == 'default':
        return_settings = {
            'generate_from_cov':True,
            'lensing_type':'simple',
            'obs_type':'spt3g_nobeam',
            'z_cluster':0.5,
            'N_pix':16,
            'pix_size_arcmin':1.0
        }
    if settings_type == 'full':
        return_settings = {
            'generate_from_cov':True,
            'lensing_type':'simple',
            'obs_type':'spt3g_nobeam',
            'z_cluster':0.5,
            'N_pix':64,
            'pix_size_arcmin':0.5
        }        
    
    return return_settings