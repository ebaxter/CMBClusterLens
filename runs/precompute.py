# run all the cosmology calculations necessary for CMB cluster lensing analysis.
# We assume that the cluster redshift will not be varied during the actual analysis.

import camb
import numpy as np
import time
import pdb
import matplotlib.pyplot as pl
from scipy import interpolate
from astropy import constants as const
from astropy import units as u
from colossus.halo import mass_so
from colossus.cosmology import cosmology as cosmo_colossus
from astropy.cosmology import FlatLambdaCDM
from camb import model
import likelihood_funcs as likelihood

def get_baseline_cosmo_params():
    params = {'H0':67.5, 'ombh2':0.022, 'omch2':0.122, 'mnu':0.06, 'omk':0, 'tau':0.06, \
              'As':2e-9, 'ns':0.965, 'r':0}
    Omega_M = (params['ombh2']+params['omch2'])/(params['H0']/100.)**2
    params['Omega_M'] = Omega_M
    return params

def get_cosmo_astropy(params):
    Omega_M = (params['ombh2']+params['omch2'])/(params['H0']/100.)**2
    cosmo_astropy = FlatLambdaCDM(params['H0'], Omega_M)
    return cosmo_astropy

def update_cosmo_params(baseline_params, z_cluster):
    #Run CAMB to get sigma8 and other cosmological parameters
    #new set of parameters for CAMB
    pars = camb.CAMBparams()
    #This function sets up with one massive neutrino and helium set using BBN consistency
    pars.set_cosmology(H0 = baseline_params['H0'], ombh2 = baseline_params['ombh2'], \
                       omch2 = baseline_params['omch2'], mnu = baseline_params['mnu'], \
                        omk = baseline_params['omk'], tau = baseline_params['tau'])
    pars.InitPower.set_params(As = baseline_params['As'], ns = baseline_params['ns'], \
                              r = baseline_params['r'])
    pars.NonLinear = model.NonLinear_none
    #Is this necessary for getting sigma8?
    pars.set_matter_power(redshifts=[0.], kmax=2.0)
                              
    results = camb.get_results(pars)
    #Get cosmological information
    camb_params = results.get_derived_params()
    #Get temperature and polarization power spectra
    #     
    chi_cluster = results.comoving_radial_distance(z_cluster)
    chi_star = results.comoving_radial_distance(camb_params['zstar'])

    pars.set_matter_power(redshifts=[0.])
    sigma8 = results.get_sigma8()
    updated_params = baseline_params.copy()
    updated_params['sigma8'] = sigma8
    updated_params['z_star']= camb_params['zstar']
    updated_params['chi_cluster'] = chi_cluster
    updated_params['chi_star'] = chi_star

    return updated_params

def set_cosmo_colossus(params):
    colossus_params = {'flat':True, 'H0':params['H0'], 'Om0':params['Omega_M'], \
                       'Ob0':params['ombh2']/(params['H0']/100.)**2, 'sigma8':params['sigma8'], 'ns':params['ns']}
    cosmo_colossus.addCosmology('myCosmo', **colossus_params)
    cosmo = cosmo_colossus.setCosmology('myCosmo')
    cosmo_colossus.setCosmology('planck18')

def get_cmblensing_window(z, chi, cosmo_params):
    a = 1./(1.+z)
    H0_units = cosmo_params['H0']*u.km/u.s/u.Mpc
    prefactor = H0_units**2*(1.5/const.c**2.)*cosmo_params['Omega_M']
    chi_star = cosmo_params['chi_star']
    q_kcmb = prefactor.to('1/Mpc**2').value*(chi/a)*(chi_star - chi)/(chi_star)
    return q_kcmb

def get_Cells(ell_max, bigell_max, z_cluster, cosmo_params, make_plots = False):
    """Compute the unlensed CMB temperature power spectrum and the lensing potential power spectrum from above and below the cluster redshift"""
    #ell is for clTT, bigell is for clkk
    bigell = np.arange(0, bigell_max+1, dtype=np.float64)

    pars = camb.CAMBparams()
    pars.NonLinear = model.NonLinear_both # what does this do?
    pars.NonLinearModel.set_params(halofit_version='takahashi')
    #This function sets up with one massive neutrino and helium set using BBN consistency
    pars.set_cosmology(H0 = cosmo_params['H0'], ombh2 = cosmo_params['ombh2'], omch2 = cosmo_params['omch2'], mnu = cosmo_params['mnu'], omk = cosmo_params['omk'], tau = cosmo_params['tau'])
    pars.InitPower.set_params(As = cosmo_params['As'], ns = cosmo_params['ns'], r = cosmo_params['r'])
    #Check accuracy settings
    pars.set_for_lmax(ell_max, lens_potential_accuracy=0)
    #Run CAMB
    results = camb.get_results(pars)
    #Get cosmological information
    camb_params = results.get_derived_params()
    #Get temperature and polarization power spectra
    #raw_cl = True means Cl are returned instead of Cl l (l+1)/2pi
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', raw_cl=True)
    
    chi_cluster = results.comoving_radial_distance(z_cluster)
    chi_star = cosmo_params['chi_star']
    
    #Get lensing power spectrum
    kmax=10  #kmax to use in CMB lensing calculation
    #Get the CMB lensing power spectrum from redshifts below the cluster redshift
    #using Limber approximation
    PK = camb.get_matter_power_interpolator(pars, nonlinear=True, 
        hubble_units=False, k_hunit=False, kmax=kmax,
        #Warning: double check  this
        var1=model.Transfer_tot,var2=model.Transfer_tot, zmax=cosmo_params['z_star'])

    nz_low = 1000
    nz_high = 1000
    #chi to integrate over for low and high z calculations
    chi_lowz = np.linspace(0.0001,chi_cluster,nz_low)
    chi_highz = np.linspace(chi_cluster, chi_star, nz_high)
    dchi_lowz = chi_lowz[1:]-chi_lowz[:-1]
    dchi_highz = chi_highz[1:]-chi_highz[:-1]
    z_lowz = results.redshift_at_comoving_radial_distance(chi_lowz)
    z_highz = results.redshift_at_comoving_radial_distance(chi_highz)
    #Get lensing window function (flat universe) CHECK THESE
    #Do Limber integral over chi
    cl_kappa_lowz =np.zeros(bigell.shape)
    cl_kappa_highz = np.zeros(bigell.shape)
    for i, l in enumerate(bigell):
        k_lowz =(l+0.5)/chi_lowz
        k_highz = (l+0.5)/chi_highz

        #Evaluate matter power spectrum at desired z, k.  Interpolation function is P(z,k)
        Pk_lowz = PK.P(z_lowz, k_lowz, grid=False)
        Pk_highz = PK.P(z_highz, k_highz, grid=False)
        #restrict to k<kmax
        above_kmax_lowz = np.where((k_lowz > kmax) | (k_lowz < 1e-4))[0]
        above_kmax_highz = np.where((k_highz > kmax) | (k_highz < 1e-4))[0]

        win_lowz = get_cmblensing_window(z_lowz, chi_lowz, cosmo_params)
        win_highz = get_cmblensing_window(z_highz, chi_highz, cosmo_params)

        integrand_lowz = Pk_lowz * win_lowz * win_lowz / chi_lowz**2
        integrand_highz = Pk_highz * win_highz * win_highz / chi_highz**2

        integrand_lowz[above_kmax_lowz] = 0
        integrand_highz[above_kmax_highz] = 0
        #Do Limber integral
        cl_kappa_lowz[i] = np.sum(0.5*(integrand_lowz[1:] + integrand_lowz[:-1])*dchi_lowz)
        cl_kappa_highz[i] = np.sum(0.5*(integrand_highz[1:] + integrand_highz[:-1])*dchi_highz)
    #CMB primary power spectra
    #Update to included polarization 777777
    clTT_unlensed=powers['unlensed_scalar'][:,0]
    clTT_lensed=powers['lensed_scalar'][:,0]
    ell = np.arange(0,len(clTT_unlensed))
    #CMB kappa power spectra
    #CAMB calculation of CMB lensing power spectrum from all z
    #CAMB outputs L^4 phi^2/2pi so to convert to kappa  need to multiply by pi/2
    cl_kappa_allz = (np.pi/2)*results.get_lens_potential_cls(bigell_max)[:,0]
    
    if (make_plots):
        #test matter power spectrum calculation
        kk = np.logspace(-4, 1, 1000)
        zz = np.zeros(1000)
        Pk_plot = PK.P(zz, kk, grid = False)
        fig, ax = pl.subplots(1,1)
        ax.plot(kk, Pk_plot*1.0e18)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('k')
        ax.set_ylabel('P(k)')
        fig.savefig('./figs/test_matter_power_spectrum.png')

        fig, ax = pl.subplots(1,1)
        ax.plot(np.arange(len(cl_kappa_allz)), 1e7*(2./np.pi)*cl_kappa_allz, label = 'CAMB all z')
        ax.plot(bigell, 1e7*(2./np.pi)*cl_kappa_lowz, label = 'low z')  
        ax.plot(bigell, 1e7*(2./np.pi)*cl_kappa_highz, label = 'high z')
        ax.plot(bigell, 1e7*(2./np.pi)*(cl_kappa_lowz+cl_kappa_highz), label = 'sum', ls = 'dashed')
        ax.set_xlim((10,2000))
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()
        ax.set_ylabel('Clpp*1e7*(L(L+1))**2/2pi')
        fig.savefig('./figs/test_clkk.png')

    return ell, clTT_unlensed, clTT_lensed, bigell, cl_kappa_highz, cl_kappa_lowz

#########################################################################################################

def get_cluster_settings(cosmo_astropy, z_cluster, cosmo_params):
    #return dictionary with cluster properties
    cluster_settings = {'z_cluster': 0.5, 'M200c': 1e14, 'c200c': 3.0}

    R200c = mass_so.M_to_R(cluster_settings['M200c'], \
                                        cluster_settings['z_cluster'], '200c')*u.kpc
    cluster_settings['R200c'] = R200c.to('Mpc')
    cluster_settings['Rs'] = cluster_settings['R200c']/cluster_settings['c200c']
    cluster_settings['dA_L'] = cosmo_astropy.angular_diameter_distance(cluster_settings['z_cluster'])
    cluster_settings['chi_L'] = cosmo_astropy.comoving_distance(cluster_settings['z_cluster'])

    #coordinates of cluster center in arcmin
    cluster_settings['cluster_center_x'] = 0.0
    cluster_settings['cluster_center_y'] = 0.0

    #Assumes flat Universe
    dA_S = cosmo_params['chi_star']/(1+cosmo_params['z_star'])*u.Mpc
    dA_L = cluster_settings['dA_L']
    chi_L = cluster_settings['chi_L']
    chi_S = cosmo_params['chi_star']*u.Mpc
    z_S = cosmo_params['z_star']
    dA_LS = (chi_S - chi_L)/(1.+z_S)
    cluster_settings['dA_LS'] = dA_LS.to('Mpc')
    cluster_settings['dA_S'] = dA_S.to('Mpc')

    #Calculate Sigma_crit
    distance_ratio = dA_S/(dA_L*dA_LS)
    sigma_crit = distance_ratio*(const.c**2/(4.*np.pi*const.G))
    cluster_settings['Sigma_crit'] = sigma_crit.to('Msun/Mpc**2')
    
    return cluster_settings

#Do all calculations to prepare for analysis, fixing cluster redshift
def prepare_analysis(z_cluster, obs_type = 'spt3g_nobeam', prep_likelihood = False, make_plots = False, N_pix = 16, pix_size_arcmin = 0.5):
    cosmo_params_baseline = get_baseline_cosmo_params()
    #Add in additional derived parameters
    cosmo_params = update_cosmo_params(cosmo_params_baseline, z_cluster)
    set_cosmo_colossus(cosmo_params)
    cosmo_astropy = get_cosmo_astropy(cosmo_params)
    
    #resolution settings for map and additional filtering
    map_settings = {'N_pix': N_pix, 'pix_size_arcmin': pix_size_arcmin, 'lmin': 2, 'lmax': 1.0e10, 'map_center_x_arcmin': 0, 'map_center_y_arcmin': 0}
    #center_x_arcmin and center_y_arcmin are coordinates of map center 
    #observational settings
    if obs_type == 'perfect':
        obs_settings = {'beam_fwhm_arcmin': 1e-9, 'noise_mukarcmin': 1.0e-10, 'lx_max': 1.0e10}
    if obs_type == 'spt3g_nobeam':
        obs_settings = {'beam_fwhm_arcmin': 1e-9, 'noise_mukarcmin': 5.0, 'lx_max': 1.0e10}
    

    #Get the CMB temperature/polarization power spectra and lensing potential power spectrum
    #Warning: what should these be?
    ell_max = 3000
    bigell_max = 2000
    ell, clTT_unlensed, clTT_lensed, bigell, clpp_highz, clpp_lowz = get_Cells(ell_max, bigell_max, z_cluster, cosmo_params)
    spectra = {'ell': ell, 'clTT_unlensed': clTT_unlensed, 'bigell': bigell, \
               'clpp_highz': clpp_highz, 'clpp_lowz': clpp_lowz}
    all_settings = {}
    all_settings['cluster_settings'] = get_cluster_settings(cosmo_astropy, z_cluster, cosmo_params)
    all_settings['map_settings'] = map_settings
    all_settings['obs_settings'] = obs_settings
    all_settings['spectra'] = spectra
    all_settings['cosmo_params'] = cosmo_params

    if (prep_likelihood):
        #Do calculations to prepare for likelihood calculation
        #build an interpolation function for the covariance as a function of angular separation
        num_table = 5000
        ang_sep_table = np.linspace(0.,300.*(1./60.)*(np.pi/180.), num_table)
        cov_element_table_unlensed = np.zeros(num_table)
        cov_element_table_lensed = np.zeros(num_table)
        for ii in range(0, num_table):
            #note that we use unlensed CMB power spectrum here 
            cov_element_table_unlensed[ii] = likelihood.get_cov_element(ell, clTT_unlensed, ang_sep_table[ii])
            cov_element_table_lensed[ii] = likelihood.get_cov_element(ell, clTT_lensed, ang_sep_table[ii])
            #Functions that let us compute covariance matrix elements
        cov_interp_func_unlensed = interpolate.interp1d(ang_sep_table, cov_element_table_unlensed)
        cov_interp_func_lensed = interpolate.interp1d(ang_sep_table, cov_element_table_lensed)

        likelihood_info = {'cov_interp_func_unlensed':cov_interp_func_unlensed, \
                           'cov_interp_func_lensed':cov_interp_func_lensed}
        print("likelihood_info computed")
        all_settings['likelihood_info'] = likelihood_info
    if (make_plots):
        print("map settings = ", map_settings)
        print("obs settings = ", obs_settings)
        fig, ax = pl.subplots(2,1)
        l_prefactor = ell*(ell+1)/(2.*np.pi)
        ax[0].plot(ell, l_prefactor*clTT_unlensed, label = 'clTT_unlensed')
        ax[0].set_ylabel('l(l+1)clTT_unlensed/2pi')
        ax[1].plot(bigell, clpp_highz, label = 'clpp_highz')
        ax[1].plot(bigell, clpp_lowz, label = 'clpp_lowz')
        ax[0].set_xscale('log')
        ax[0].set_yscale('log')
        ax[1].set_xscale('log')
        #ax[1].set_yscale('log')
        ax[0].legend()
        ax[1].legend()
        fig.savefig('./figs/power_spectra.png')

    return all_settings

if __name__ == '__main__':
    t1 = time.time()
    all_settings = prepare_analysis(0.5, make_plots = True)
    t2 = time.time()
    print("It took {} seconds to run prepare_analysis".format(t2-t1))