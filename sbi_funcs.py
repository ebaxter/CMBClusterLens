import torch
import numpy as np
import pdb

import matplotlib.pyplot as pl

def get_summary(data, summary_type = 'none'):
    N_pix = data.shape[1]
    N_sims = data.shape[0]
    if (summary_type == 'none'):
        compressed_data = data.reshape((N_sims, N_pix*N_pix))
    if (summary_type == 'meanvar'):
        #mean and variance of central 10x10 window
        compressed_data = np.zeros((N_sims, 2))
        central_window_size = 16
        min_index = int(N_pix/2) - int(central_window_size/2)
        max_index = int(N_pix/2) + int(central_window_size/2)
        compressed_data[:, 0] = np.mean(data[:,min_index:max_index,min_index:max_index], axis = (1,2))
        compressed_data[:, 1] = np.var(data[:,min_index:max_index,min_index:max_index], axis = (1,2))
    return compressed_data

def get_sbi_posterior_plots(likelihood_data, param_type, sbi_posterior,  summary_type = 'none', pcs = None, param_scaling = np.array([1.0, 1.0])):
    #Make a bunch of plots comparing the SBI posterior to the exact likelihood
    #param_type = 'mass' or 'map'
    #to compare to likelihood data, need to pass likelihood_data
    
    #Information from exact likelihood runs    
    num_trials = len(likelihood_data['lnlike_mat_list'])
    M200c_arr = likelihood_data['M200c_arr']
    c200c_arr = likelihood_data['c200c_arr']
    massparam_range = ((np.min(M200c_arr), np.max(M200c_arr)), (np.min(c200c_arr), np.max(c200c_arr)))    
    all_lnlike_mat = np.asarray(likelihood_data['lnlike_mat_list'])
    unlensed_cmb_maps = likelihood_data['unlensed_map_list']

    #Define grid for stacked SBI posterior
    num_grid_M200c = 10
    num_grid_c200c = 5
    all_sbi_lndensity = np.zeros((num_trials, num_grid_M200c, num_grid_c200c))
    #loop over all trials, storing sbi posterior for each
    num_posterior_draws_single = 5000
   
    #Individual posteriors
    #calculate posterior for an individual data set using SBI for the datasets we used to calculate exact likelihood
    num_trials = len(likelihood_data['obs_map_list'])
    for triali in range(0,num_trials):
        #observed data
        x_i = likelihood_data['obs_map_list'][triali].flatten()
        #Generate posterior samples conditioned on observed data

        if (param_type == 'map'):
            posterior_samples_unscaled = sbi_posterior.sample((num_posterior_draws_single,), x=x_i)
            #correct for parameter scaling
            temp_scaling = torch.ones(posterior_samples_unscaled.shape[1])
            temp_scaling[-2:] = torch.from_numpy(param_scaling)
            scaling_mat = torch.tile(temp_scaling, (num_posterior_draws_single,1))
            posterior_samples = posterior_samples_unscaled*scaling_mat

            #Caculate posterior mean
            posterior_mean = torch.mean(posterior_samples, axis = 0)
            posterior_mean_pcs = posterior_mean[:-2] #exclude the mass and concentration parameters
            posterior_samples_Mc = posterior_samples[:,-2:] #mass and concentration parameters
            #Convert posterior mean pcs into posterior mean unlensed CMB map
            N_pix = x_i.shape[0]
            N_side = int(np.sqrt(N_pix))
            posterior_mean_map = np.matmul(posterior_mean_pcs, pcs.T).reshape(N_side,N_side)

            #The true unlensed map
            map_unlensed_i = unlensed_cmb_maps[triali]
            #The true unlensed map as represented by the PCs
            pc_amplitudes = np.dot(map_unlensed_i.flatten(), pcs) 
            map_unlensed_i_pc = np.matmul(pc_amplitudes, pcs.T).reshape(N_side,N_side)
        elif (param_type == 'mass'):
            posterior_samples_Mc = posterior_samples


    if (param_type == 'map'):
        #Plot posterior mean map next to true unlensed CMB map
        fig, ax = pl.subplots(4,1, figsize = (4,14))
        ax[0].imshow(posterior_mean_map)
        ax[0].set_title('posterior mean')
        ax[1].imshow(map_unlensed_i)
        ax[1].set_title('true unlensed CMB map')
        ax[2].imshow(map_unlensed_i_pc)
        ax[2].set_title('true unlensed CMB map from PCs')
        true_data = x_i.reshape(N_side,N_side)
        ax[3].imshow(true_data)
        ax[3].set_title('Observed data')
        fig.savefig('./figs/posterior_sample_map_{}.png'.format(triali))

    #Individual posterior on M and c
    fig_indiv, ax_indiv = pl.subplots(1,1)
    ax_indiv.contourf(c200c_arr, M200c_arr, all_lnlike_mat[triali,:,:], cmap='Greens')
    ax_indiv.scatter(posterior_samples_Mc[:,1], posterior_samples_Mc[:,0], marker = 'x', color = 'red')
    ax_indiv.set_xlabel('c200c')
    ax_indiv.set_ylabel('M200c')
    ax_indiv.set_xlim(c200c_arr[0], c200c_arr[-1])
    ax_indiv.set_ylim(M200c_arr[0], M200c_arr[-1])
    fig_indiv.savefig('./figs/posterior_indiv_sample_Mc_paramtype{}_{}.png'.format(param_type, triali))

    #Stacked posteriors
    print("Computing stacked posterior\n")
    #Each trial is a different mock data set
    #Build stacked SBI posterior from all trials
    num_posterior_draws_stack = 15000
    for triali in range(0,num_trials):
        print("trial = ", triali)
        true_params = likelihood_data['true_params']  #note that for now, true params are always the same
        datai = likelihood_data['obs_map_list'][triali]
        datai = np.expand_dims(datai, axis = 0)
        lnlike_mati = likelihood_data['lnlike_mat_list'][triali]
        #generate posterior samples from SBI model
        x0 = torch.tensor(get_summary(datai, summary_type = summary_type), dtype=torch.float32)
        posterior_samples_unscaled = sbi_posterior.sample((num_posterior_draws_stack,), x=x0)

        temp_scaling = torch.ones(posterior_samples_unscaled.shape[1])
        temp_scaling[-2:] = torch.from_numpy(param_scaling)
        scaling_mat = torch.tile(temp_scaling, (num_posterior_draws_stack,1))
        posterior_samples = posterior_samples_unscaled*scaling_mat

        mass_samples = posterior_samples[:,-2:].numpy()
        pc_samples = posterior_samples[:,:-2].numpy()

        #Compute density across grid from posterior samples
        bins = (num_grid_M200c,num_grid_c200c) #(num_M200c, num_c200c)
        M200c_samples = mass_samples[:,0]
        c200c_samples = mass_samples[:,1]
        sbi_density, M200c_edges, c200c_edges = np.histogram2d(M200c_samples, c200c_samples, bins = bins, range = massparam_range, density = True)
        #store SBI density for this trial
        all_sbi_lndensity[triali,:,:] = np.log(sbi_density)

    #Stacked contour plot
    fig_stacked_contours, ax_stacked_contours = pl.subplots(1,1, figsize = (8,6))
    stacked_lnlike = np.sum(all_lnlike_mat[0:num_trials,:,:], axis = 0)
    ax_stacked_contours.contourf(c200c_arr, M200c_arr, stacked_lnlike, cmap='Greens')

    #overplot stacked sbi density
    stacked_sbi_lndensity = np.sum(all_sbi_lndensity, axis = 0)
    c200c_cents = (c200c_edges[1:] + c200c_edges[:-1])/2.
    M200c_cents = (M200c_edges[1:] + M200c_edges[:-1])/2.
    ax_stacked_contours.contour(c200c_cents, M200c_cents, stacked_sbi_lndensity, cmap='Reds')

    #add true parameter point
    ax_stacked_contours.plot(true_params[1], true_params[0], 'x', markersize=10, label='True', color = 'lightblue')
    ax_stacked_contours.set_xlabel('c200c')
    ax_stacked_contours.set_ylabel('M200c')

    #match axis range to likelihood contours
    ax_stacked_contours.set_xlim(c200c_arr[0], c200c_arr[-1])
    ax_stacked_contours.set_ylim(M200c_arr[0], M200c_arr[-1])
    fig_stacked_contours.savefig('./figs/' + 'stacked_sbi_and_likelihood_paramtype{}.png'.format(param_type))
    pdb.set_trace()
