import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np 
from utils import *
from ADMM import *

data_dir = '../datasets/PEMS0X_data'
# TODO: change here
dataset = 'PEMS04'
data_folder = os.path.join(data_dir, dataset)
data_file = dataset + '.npz'
graph_csv = dataset + '.csv'
# data
traffic_dataset = TrafficDataset(data_folder, data_file, graph_csv, transform='normalize')
print(f"data shape: {traffic_dataset.data.shape}, node number: {traffic_dataset.graph_info['n_nodes']}, edge number: {traffic_dataset.graph_info['n_edges']}")

# kNNs and graph construction
k = 4
nearest_nodes, nearest_dists = k_nearest_neighbors(traffic_dataset.graph_info['n_nodes'], traffic_dataset.graph_info['u_edges'], traffic_dataset.graph_info['u_dist'], k)
print(f'nearest nodes: {nearest_nodes.shape}, nearest_dists: {nearest_dists.shape}')

# mixed_graph_from_distance()

x, y = traffic_dataset.get_predict_data(0)
x, y = x.unsqueeze(0), y.unsqueeze(0)
print(f'recovering: x: {x.shape}, y: {y.shape}')

# difference in data
diff_x, diff_y = get_data_difference(x), get_data_difference(y)
print(f'difference: x: {diff_x.shape}, y: {diff_y.shape}')

# interpolation
interp_x, interp_y, mask = traffic_dataset.get_interpolated_data(0, 0.4)
interp_x, interp_y, mask = interp_x.unsqueeze(0), interp_y.unsqueeze(0), mask.unsqueeze(0)
print(f'interpolation: x: {interp_x.shape}, y: {interp_y.shape}, mask: {mask.shape}')