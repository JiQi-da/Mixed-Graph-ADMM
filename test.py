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
k = 6
nearest_nodes, nearest_dists = k_nearest_neighbors(traffic_dataset.graph_info['n_nodes'], traffic_dataset.graph_info['u_edges'], traffic_dataset.graph_info['u_dist'], k)
print(f'nearest nodes: {nearest_nodes.shape}, nearest_dists: {nearest_dists.shape}')

# mixed_graph_from_distance()

x, y = traffic_dataset.get_data(0)
print(f'training shape: x: {x.shape}, y: {y.shape}')

# %%
# test primal guess
x, y = x.unsqueeze(0), y.unsqueeze(0)
print(x.dtype, y.dtype)

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
                            k=7,
                            u_sigma=50,
                            d_sigma=50,
                            # ablation='DGTV'
                            )

# %%
# test ADMM algorithm
x_pred = admm_block.combined_loop(y)
# admm_block.plot_residual()

