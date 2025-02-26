import networkx as nx
import pandas as pd
import os
from collections import Counter
import heapq
import torch
import numpy as np

def physical_graph(df, sensor_dict=None):
    if sensor_dict is None:
        from_list, to_list = list(df['from'].values), list(df['to'].values)
    else:
        from_list = [sensor_dict[i] for i in df['from'].values]
        to_list = [sensor_dict[i] for i in df['to'].values]
    n_edges = len(from_list)
    # bi-directional
    u_edges = torch.tensor([from_list + to_list, to_list + from_list]).T
    dic = Counter([(u_edges[i,0].item(), u_edges[i,1].item()) for i in range(n_edges)])
    assert max(list(dic.values())), 'distance graph asymmetric'
    ew1 = torch.tensor(df[df.columns[-1]].values)
    u_distance = torch.cat([ew1, ew1])
    return n_edges, u_edges, u_distance

class TrafficDataset():
    def __init__(self, data_folder, data_file, graph_csv, id_file=None):
        graph_path = os.path.join(data_folder, graph_csv)
        self.df = pd.read_csv(graph_path, index_col=None)
        if id_file is not None:
            id_path = os.path.join(data_folder, id_file)
            sensor_id = torch.tensor(np.loadtxt(id_path, dtype=int))
            n_nodes = sensor_id.shape[0]
            sensor_dict = {sensor_id[k].item(): k for k in range(n_nodes)}
        else:
            n_nodes = max(max(self.df['from'].values), max(self.df['to'].values)) + 1
            sensor_dict = None
        
        n_edges, u_edges, u_distance = physical_graph(self.df, sensor_dict)

        self.graph_info = {
            'n_nodes': n_nodes,
            'n_edges': n_edges,
            'u_edges': u_edges,
            'u_dist': u_distance
        }

        self.data = torch.tensor(np.load(os.path.join(data_folder, data_file))['data'][...,:1]) # in (T, N, 1)

        # TODO: normalize data
    
    def get_predict_data(self, index):
        x = self.data[index:index + 24]
        y = self.data[index:index + 12]
        return x, y
    
    def get_interpolated_data(self, index, mask_rate=0.4):
        x = self.data[index:index + 24]
        torch.manual_seed(42)
        mask = (torch.rand_like(x) >= mask_rate).float()
        y = x * mask
        # y[mask == 0] = float('nan')
        # print(f'Mask rate: {mask_rate}')
        assert not torch.isnan(y[mask == 1]).any(), "Masked data should not be nan"
        return x, y, mask

    def get_databatch(self, index, batch_size):
        X, Y = []
        for i in range(batch_size):
            x, y = self.get_data(index + i)
            X.append(x)
            Y.append(y)
        return torch.tensor(X), torch.tensor(Y)

def get_data_difference(data:torch.Tensor):
    '''
    data: (B, T, N, 1)
    Return: (B, T, N, 1)
    '''
    assert data.ndim == 4, "Data should have 4 dims (B, T, N, C)"
    # y = data.clone()
    y = data[:,1:] - data[:,:-1]
    # y[0] = 0
    return y


def connect_list(n_nodes, edges, dists):
    '''
    return (N, k) where k is the maximum degree
    '''
    counts = torch.zeros(n_nodes, dtype=torch.int)
    for edge in edges:
        counts[edge[0]] += 1
    k = counts.max().item()
    print(counts)
    print('max degrees', k)

    connect_list = -torch.ones((n_nodes, k + 1), dtype=torch.int)
    dist_list = torch.full((n_nodes, k + 1), float('inf'))
    for i in range(len(edges)):
        connect_list[edges[i, 0], counts[edges[i, 0]]] = edges[i, 1]
        dist_list[edges[i, 0], counts[edges[i, 0]]] = dists[i]
        counts[edges[i, 0]] -= 1
    
    assert torch.all(counts == 0), "Counts should be a zero matrix after processing all edges"
    assert torch.all(connect_list[:,0] == -1), "connect list should be all -1 in the first row when not finished"
    connect_list[:,0] = torch.arange(n_nodes)
    dist_list[:,0] = torch.zeros(n_nodes)

    connect_list = connect_list.to(torch.long)

    return connect_list, dist_list # in (N, k)

def k_nearest_neighbors(n_nodes, edges:torch.Tensor, dists:torch.Tensor, k):
    '''
    -----------------------------
    Return:
    - kNN neighbors for each node: (N, k + 1), -1 for non-existing
    - kNN distances for each node: (N, k + 1), -1 for non-existing (or inf?)
    '''
    graph = nx.DiGraph()
    for i in range(len(edges)):
        graph.add_edge(edges[i,0].item(), edges[i,1].item(), weight=dists[i].item())
    print(f'{n_nodes} nodes, {k} neighbors')
    # holder
    nearest_nodes = - torch.ones((n_nodes, k + 1), dtype=torch.int)
    nearest_dists = torch.full((n_nodes, k + 1), float('inf'))

    for node in range(n_nodes):
        distances = nx.single_source_dijkstra_path_length(graph, node)
        closest_nodes = heapq.nsmallest(k + 1, distances.items(), key=lambda x: x[1])
        k_true = len(closest_nodes)
        nearest_nodes[node,:k_true] = torch.tensor([i for (i,_) in closest_nodes])
        nearest_dists[node,:k_true] = torch.tensor([j for (_,j) in closest_nodes])
    return nearest_nodes, nearest_dists

def undirected_graph_from_distance(connect_list:torch.Tensor, dist_list:torch.Tensor, u_sigma=None, regularized=True):
    '''
    connect_list: connection from each node to the other (include a self-loop in the 0th entry, pad -1 as placeholder)

    dist_list: distance list from each node to the other
    sigma: scalar, if None, sigma = max(dists) / 100
    -----------------------------------
    Return:
    Single head only, can be expanded to multi-head later
    - Directed graph weights (N, k), can be done with a different sigma ?
    - Undirected graph weights (N, k)
    The difference:(i,t) -> (i, t+1) in directed graph, but no self loop in directed graphs
    '''
    n_nodes = connect_list.shape[0]
    dist_mask = (connect_list != -1) & (dist_list != 0)
    dist_values = dist_list[dist_mask]
    if u_sigma == None:
        u_sigma = max(dist_values.max().item() / 50, dist_values.min().item() * 50)
    print(f'Undirected graph: sigma = {u_sigma}, nearest_dist in ({dist_values.min().item():.4f}, {dist_values.max().item():.4f})')
    
    weights = torch.exp(- dist_list[:,1:] / u_sigma) # in (N, k)
    # mask where there's no connection
    zero_mask = (connect_list[:,1:] == -1)
    weights[zero_mask] = 0

    if regularized:
        # NOTICE: regularize ONLY to each node's connection (leave out the case where i is j's kNN but j is not i's kNN)
        degree = weights.sum(1) # in (N)
        degree_j = degree[connect_list[:, 1:]].reshape(n_nodes, -1) # in (N, k)
        degree_ij = degree.unsqueeze(1) * degree_j
        inv_sqrt_degree_ij = torch.where(degree_ij > 0, 1 / torch.sqrt(degree_ij), torch.zeros_like(degree_ij))
        weights = weights * inv_sqrt_degree_ij
    return weights

def directed_graph_from_distance(connect_list:torch.Tensor, dist_list:torch.Tensor, d_sigma=None, regularized=True):

    n_nodes = connect_list.shape[0]
    dist_mask = (connect_list != -1) & (dist_list != 0)
    dist_values = dist_list[dist_mask]
    if d_sigma == None:
        d_sigma = max(dist_values.max().item() / 50, dist_values.min().item() * 50)
    print(f'Directed Graph: sigma = {d_sigma}, nearest_dist in ({dist_values.min().item():.4f}, {dist_values.max().item():.4f})')
    
    weights = torch.exp(- dist_list / d_sigma) # in (N, k + 1)
    zero_mask = (connect_list == -1)
    weights[zero_mask] = 0

    if regularized:
        # regularize on each node's children
        in_degree = weights.sum(1) # in (T, N, k)
        inv_in_degree = torch.where(in_degree > 0, 1 / in_degree, torch.zeros_like(in_degree))
        weights = weights * inv_in_degree.unsqueeze(1)
    return weights

def recover_Laplacians(connect_list, weights):
    '''
    For undirected graphs only, on each time slice
    connect_list: (N, k)
    weights: (N, k)
    Return:
    - L: (N, N)
    '''
    n_nodes = connect_list.shape[0]
    L = torch.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        L[i, i] = weights[i].sum()
        for j in connect_list[i]:
            if j != -1:
                L[i, j] = - weights[i, j]
    # compute condition numbers
    eigvals = torch.symeig(L, eigenvectors=False).eigenvalues
    cond = eigvals.max() / eigvals.min()
    print(f'Condition number: {cond:.4f}')
    return L, cond


def line_graph(n_nodes):
    '''
    edges: (n_edges, 2)
    dists: (n_edges)
    Return: 
    - connect_list in (n_nodes, k) where k = 1
    - weights in (n_nodes, k) where k = 1
    '''
    connect_list = torch.arange(n_nodes).unsqueeze(1)
    weights = torch.ones_like(connect_list).float()
    return connect_list, weights

def expand_time_dimension(u_ew, d_ew, T:int):
    return u_ew.unsqueeze(0).repeat(T, 1, 1), d_ew.unsqueeze(0).repeat(T - 1, 1, 1)

if __name__ == "__main__":
    edges = torch.tensor([[0,1], [1,2], [2,3], [3,2], [2,1], [1,0]], dtype=torch.int)
    dists = torch.tensor([1,2,3,3,2,1])
    print(connect_list(4, edges, dists))
    # 其他测试代码...

# TODO: to be constructed with our Unrolling version
class MixedGraphFromFeatures():
    def __init__(self,):
        pass
    
    def undirected_graph_weights(self,):
        pass

    def directed_graph_weights(self,):
        pass

    def get_weights(self,):
        pass