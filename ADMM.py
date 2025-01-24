import numpy as np
import torch
import os
from utils import *


class ADMM_algorithm():
    def __init__(self):
        pass
    # operators
    def apply_op_Lu(self, x):
        pass
    
    def apply_op_Ld(self, x):


data_dir = '../datasets/PEMS0X_data'
# change here
dataset = 'PEMS04'
data_folder = os.path.join(data_dir, dataset)
data_file = dataset + '.npz'
graph_csv = dataset + '.csv'
# data
traffic_dataset = TrafficDataset(data_folder, data_file, graph_csv)
print(f'data shape: {traffic_dataset.data.shape}, edges number: {traffic_dataset.n_edges}, edges shape: {traffic_dataset.u_edges.shape}')

# kNNs and graph construction
k = 4
nearest_nodes, nearest_dists = k_nearest_neighbors(traffic_dataset.n_nodes, traffic_dataset.u_edges, traffic_dataset.u_distance, k)

mixed_graph_from_distance()

x, y = traffic_dataset.get_data(0)
print('training shape', x.shape, y.shape)

# graph construction