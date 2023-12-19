import matplotlib.pyplot as pl
import numpy as np
import pickle as pk

likelihood_grid_filename = 'likelihood_grid_Npix32_generatefromcovTrue_usepcsFalse_1130_num5.pk'
likelihood_data = pk.load(open('./likelihood_grids/' + likelihood_grid_filename, 'rb'))
M200c_arr = likelihood_data['M200c_arr']
c200c_arr = likelihood_data['c200c_arr']
num_M200c = len(M200c_arr)
num_c200c = len(c200c_arr)

#number of mock data sets that have likelihood grids
num_trials = len(likelihood_data['lnlike_mat_list'])

#Compute and plot stacked likelihood
all_lnlike_mat = np.asarray(likelihood_data['lnlike_mat_list'])
stacked_lnlike_mat = np.sum(all_lnlike_mat, axis = 0)
fig_stack, ax_stack = pl.subplots(1,1, figsize = (8,6))
levels = np.max(stacked_lnlike_mat) - np.array([4., 3., 2., 1., 0.])
ax_stack.contourf(c200c_arr, M200c_arr, stacked_lnlike_mat, cmap='Blues', levels = levels)
true_params = likelihood_data['true_params']  #note that for now, true params are always the same
ax_stack.plot(true_params[1], true_params[0], marker = 'x', markersize=10, label='True', color = 'red')
ax_stack.set_xlabel('c200c')
ax_stack.set_ylabel('M200c')
fig_stack.savefig('./figs/stacked_lnlike_{}.png'.format(likelihood_grid_filename))