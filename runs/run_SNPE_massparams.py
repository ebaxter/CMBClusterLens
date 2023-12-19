import numpy as np
import pdb
import pickle as pk
import torch
from sbi import utils as utils
from sbi.inference import SNPE, prepare_for_sbi
from sbi.inference.base import infer


#for embedding net
import torch.nn as nn
import torch.nn.functional as F

import sbi_funcs as sbi_funcs

#plotting
from getdist import plots, MCSamples
import matplotlib.pyplot as pl

summary_type = 'none'

#load simulation data for training SBI
data_filename = 'sims_lensingtypesimple_scaled_generatefromcovTrue_Nsims2000_Npix32_1202.npz'
sim_data = np.load('./sims/' + data_filename, allow_pickle=True)
#parameter values for each simulation
params = sim_data['params']
N_sims = params.shape[0]
#generate summary statistics for all data sets
data = sbi_funcs.get_summary(sim_data['data'], summary_type = summary_type)
param_min = sim_data['param_min']
param_max = sim_data['param_max']
param_scaling = sim_data['param_scaling']
#parameters corresponding to each simulation
theta = torch.tensor(sim_data['params'], dtype=torch.float32)

#SNPE #############################################################################################################################
#Train the SBI model
#set up sbi priors
prior = utils.BoxUniform(low=param_min, high=param_max)

device = 'cpu'
neural_posterior = utils.posterior_nn(model="maf", hidden_features=10, num_transforms=2, device=device)

#For embedding net
class SummaryNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 2D convolutional layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        # Maxpool layer that reduces 32x32 image to 4x4
        self.pool = nn.MaxPool2d(kernel_size=8, stride=8)
        # Fully connected layer taking as input the 6 flattened output arrays from the maxpooling layer
        self.fc = nn.Linear(in_features=6 * 4 * 4, out_features=8)

    def forward(self, x):
        x = x.view(-1, 1, 32, 32)
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 6 * 4 * 4)
        x = F.relu(self.fc(x))
        return x

#See https://sbi-dev.github.io/sbi/tutorial/05_embedding_net/
embedding_net = SummaryNet()
# instantiate the neural density estimator

neural_posterior = utils.posterior_nn(
    model="maf", embedding_net=embedding_net, hidden_features=10, num_transforms=2
)

inference = SNPE(prior=prior,density_estimator=neural_posterior)
x = torch.tensor(data, dtype=torch.float32)
inference.append_simulations(theta, x)
density_estimator = inference.train(max_num_epochs=1000)
posterior = inference.build_posterior(density_estimator)

#Compare to likelihood calculation
#load likelihood grid data (mock data sets, true parameter values, and corresponding likelihood grids in M200, c200 space)
likelihood_grid_filename = 'likelihood_grid_Npix32_generatefromcovTrue_usepcsFalse_1130_num5.pk'
likelihood_data = pk.load(open('./likelihood_grids/' + likelihood_grid_filename, 'rb'))
all_lnlike_mat = np.asarray(likelihood_data['lnlike_mat_list'])
M200c_arr = likelihood_data['M200c_arr']
c200c_arr = likelihood_data['c200c_arr']
num_M200c = len(M200c_arr)
num_c200c = len(c200c_arr)

#Generate plots of SBI posteriors and comparison to exact likelihood
#sbi_funcs.get_sbi_posterior_plots(likelihood_data, 'mass', posterior, param_scaling=param_scaling)