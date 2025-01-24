import numpy as np
import torch
import os
from utils import *


class ADMM_algorithm():
    '''
    only with 1 head
    '''
    def __init__(self, graph_info, ADMM_info, use_kNN=False, k=4, u_sigma=None, d_sigma=None, expand_time_dim=True):
        self.n_nodes = graph_info['n_nodes']
        self.u_edges = graph_info['u_edges']
        self.u_dists = graph_info['u_dist']
        self.use_kNN = use_kNN
        # TODO: further we can use different connection in directed graphs and undirected graphs
        if use_kNN:
            self.connect_list, self.dist_list = k_nearest_neighbors(self.n_nodes, self.u_edges, self.u_dists, k)
            self.connect_list = self.connect_list.to(torch.int64)
        else:
            self.connect_list, self.dist_list = connect_list(self.n_nodes, self.u_edges, self.u_dists)
        
        self.u_ew = undirected_graph_from_distance(self.connect_list, self.dist_list, sigma=u_sigma, regularized=True)
        self.d_ew = undirected_graph_from_distance(self.connect_list, self.dist_list, sigma=u_sigma, regularized=True)

        if expand_time_dim:
            self.u_ew, self.d_ew = expand_time_dimension(self.u_ew, self.d_ew) # in (T, N, k)

    # operators
    def apply_op_Lu(self, x):
        '''
        signal shape: (B, T=24, N, n_channels)
        '''
        B, T, n_channels = x.size(0), x.size(1), x.size(-1)
        pad_x = torch.zeros_like(x[:,:,0:1])
        pad_x = torch.cat((x, pad_x), 2)

        # gather feature on each signals
        weights_features = self.u_ew.unsqueeze(0).unsqueeze(-1) * pad_x[:,:,self.connect_list[:,1:].reshape(-1)].reshape(B, T, self.n_nodes, -1, n_channels)
        return x - weights_features.sum(3)
    
    def apply_op_Ldr(self, x):
        B, T, n_channels = x.size(0), x.size(1), x.size(-1)
        pad_x = torch.zeros_like(x[:,:,0:1])
        pad_x = torch.cat((x, pad_x), 2)
        # gather features on each children
        child_features = self.d_ew.unsqueeze(0).unsqueeze(-1) * pad_x[:, :-1, self.connect_list.view(-1)].view(B, T-1, self.n_nodes, -1, n_channels)
        x[:,1:] = x[:,1:] - child_features.sum(3)
        # time 0 has no children
        x[:,0] = x[:,0] * 0
        return x
    
    def apply_op_Ldr_T(self, x:torch.tensor):
        B ,T, n_channels = x.size(0), x.size(1), x.size(-1)
        if self.use_kNN:
            # assymetric graph constructions with kNN. Compute weighted features for each father, scatter-add to the childrens

            holder = self.d_ew.unsqeeze(0).unsqueeze(-1) * x[:,1:].unsqueeze(3) # (B, T-1, N, k, n_channels)
            father_features = torch.zeros((B, T-1, self.n_nodes+1, n_channels))
            index = self.connect_list.reshape(-1)[None, None, :, None].repeat(B, T-1, 1, n_channels)
            index[index == -1] = self.n_nodes
            if torch.any(index < 0) or torch.any(index >= father_features.size(2)):
                raise ValueError("Index out of bounds")
            father_features = father_features.scatter_add(2, index, holder.view(B, T-1, -1, n_channels))
        else:
            # the graph connect list is symmetric. Direct apply on the father nodes
            pad_x = torch.zeros_like(x[:,:,0:1])
            pad_x = torch.cat((x, pad_x), 2)
            father_features = self.d_ew.unsqueeze(0).unsqueeze(-1) * pad_x[:, 1:, self.connect_list.view(-1)].view(B, T-1, self.n_nodes, -1, n_channels)

        # the end nodes (in time T) have no children, y[:, -1] = x
        # the source nodes in time 0 have no fathers, y[:,0] = - father_features[:,0]
        # other nodes: y[:,t] = x[:,t] - father_features[:, t]
        # y = x.clone()
        x[:,0] = x[:,0] * 0
        x[:,:-1] = x[:, :-1] - father_features.sum(3)
        return y
    
    def apply_op_cLdr(self, x):
        y = self.apply_op_Ldr(x)
        y = self.apply_op_Ldr_T(x)
        return y
    
    



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