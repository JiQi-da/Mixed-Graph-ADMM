# %%
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np 
from utils import *
from ADMM import *

# %%
data_dir = '../datasets/PEMS0X_data'
# TODO: change here
dataset = 'PEMS04'
data_folder = os.path.join(data_dir, dataset)
data_file = dataset + '.npz'
graph_csv = dataset + '.csv'
# data
traffic_dataset = TrafficDataset(data_folder, data_file, graph_csv)
print(f"data shape: {traffic_dataset.data.shape}, node number: {traffic_dataset.graph_info['n_nodes']}, edge number: {traffic_dataset.graph_info['n_edges']}")

# kNNs and graph construction
k = 4
nearest_nodes, nearest_dists = k_nearest_neighbors(traffic_dataset.graph_info['n_nodes'], traffic_dataset.graph_info['u_edges'], traffic_dataset.graph_info['u_dist'], k)
print(f'nearest nodes: {nearest_nodes.shape}, nearest_dists: {nearest_dists.shape}')

# mixed_graph_from_distance()

x, y = traffic_dataset.get_data(0)
print(f'training shape: x: {x.shape}, y: {y.shape}')

# %%
# test primal guess
x, y = x.unsqueeze(0), y.unsqueeze(0)
print(x.dtype, y.dtype)
print(torch.arange(0,12).dtype)
x_init = initial_guess(y, 12, 24)
print(x_init.shape)

# %%
# plot primal guess
from matplotlib import pyplot as plt

x_top = x_init[:,:,0:10].squeeze()
print(x_top.shape)
t = torch.arange(0,24,1)
print(t.shape)
plt.figure()
plt.plot(t, x_top)
plt.legend([f'node {i}' for i in range(10)])
plt.title('Linear regression for initial guess')
plt.show()

# %%
import math
rho_init = math.sqrt(traffic_dataset.graph_info['n_nodes'] / 24)
print('rho_init:', rho_init)
ADMM_info = {
    'rho': rho_init,
    'rho_u': rho_init,
    'rho_d': rho_init,
    'mu_u': 3,
    'mu_d1':3,
    'mu_d2': 3
}
admm_block = ADMM_algorithm(traffic_dataset.graph_info,
                            ADMM_info,
                            use_kNN=True,
                            k=4,
                            u_sigma=20,
                            d_sigma=20,
                            )


