import networkx as nx
import numpy as np
import pandas as pd
import os
from collections import Counter
import heapq

def physical_graph(df, sensor_dict=None):
    if sensor_dict is None:
        from_list, to_list = list(df['from'].values), list(df['to'].values)
    else:
        from_list = [sensor_dict[i] for i in df['from'].values]
        to_list = [sensor_dict[i] for i in df['to'].values]
    n_edges = len(from_list) * 2
    # bi-directional
    u_edges = np.array([from_list + to_list, to_list + from_list]).T
    dic = Counter([(u_edges[i,0], u_edges[i,1]) for i in range(n_edges)])
    assert max(list(dic.values())), 'distance graph asymmetric'
    ew1 = df[df.columns[-1]].values
    u_distance = np.stack([ew1, ew1]).reshape(-1)
    return n_edges, u_edges, u_distance

class TrafficDataset():
    def __init__(self, data_folder, data_file, graph_csv, id_file=None):
        graph_path = os.path.join(data_folder, graph_csv)
        self.df = pd.read_csv(graph_path, index_col=None)
        if id_file is not None:
            id_path = os.path.join(data_folder, id_file)
            sensor_id = np.loadtxt(id_path, dtype=int)
            self.n_nodes = sensor_id.shape[0]
            self.sensor_dict = dict([sensor_id[k], k] for k in range(self.n_nodes))
        else:
            self.n_nodes = max(max(self.df['from'].values), max(self.df['to'].values)) + 1
            self.sensor_dict = None
        
        self.n_edges, self.u_edges, self.u_distance = physical_graph(self.df, self.sensor_dict)

        self.data = np.load(os.path.join(data_folder, data_file))['data'][...,:1] # in (T, N, 1)
    
    def get_data(self, index):
        x = self.data[index:index + 24]
        y = self.data[index:index + 12]
    

def k_nearest_neighbors(n_nodes, edges:np.ndarray, dists:np.ndarray, k):
    '''
    -----------------------------
    Return:
    - kNN neighbors for each node: (N, k + 1), -1 for non-existing
    - kNN distances for each node: (N, k + 1), -1 for non-existing (or inf?)
    '''
    graph = nx.DiGraph()
    for i in range(len(edges)):
        graph.add_edge(edges[i,0], edges[i,1], weight=dists[i])
    print(f'{n_nodes} nodes, {k} neighbors')
    # holder
    nearest_nodes = - np.ones((n_nodes, k + 1), dtype=int)
    nearest_dists = np.full((n_nodes, k + 1), np.inf)

    for node in range(n_nodes):
        distances = nx.single_source_dijkstra_path_length(graph, node)
        closest_nodes = heapq.nsmallest(k + 1, distances.item(), key=lambda x: x[1])
        k_true = len(closest_nodes)
        nearest_nodes[:,:k_true] = np.array([i for (i,_) in closest_nodes])
        nearest_dists[:,:k_true] = np.array([j for (_,j) in closest_nodes])
    return nearest_nodes, nearest_dists


def mixed_graph_from_distance(nearest_nodes:np.ndarray, nearest_dists:np.ndarray, u_sigma=None, d_sigma=None):
    '''
    sigma: scalar, if None, sigma = max(dists) / 100
    -----------------------------------
    Return:
    Single head only, can be expanded to multi-head later
    - Directed graph weights (N, k), can be done with a different sigma ?
    - Undirected graph weights (N, k)
    The difference:(i,t) -> (i, t+1) in directed graph, but no self loop in directed graphs
    '''
    if u_sigma == None:
        u_sigma = max(nearest_dists.max() / 50, nearest_dists.min() * 50)
        print(f'u_sigma = {u_sigma}, nearest_dist in ({nearest_dists.min():.4f}, {nearest_dists.max():.4f})')
    
    if d_sigma == None:
        d_sigma = max(nearest_dists.max() / 50, nearest_dists.min() * 50)
        print(f'd_sigma = {d_sigma}, nearest_dist in ({nearest_dists.min():.4f}, {nearest_dists.max():.4f})')
    
    undirected_weights = np.exp(- nearest_dists / u_sigma) # in (N, k + 1)
    directed_weights = np.exp(- nearest_dists / d_sigma) # in (N, k + 1)
    zero_mask = nearest_nodes[nearest_nodes == -1]
    undirected_weights[zero_mask] = 0
    directed_weights[zero_mask] = 0
    undirected_weights = undirected_weights[:, 1:] # in (N, k)

    return undirected_weights, directed_weights # in (N, k), (N, k+1)



class MixedGraphFromFeatures():
    def __init__(self,):
        pass
    
    def undirected_graph_weights(self,):
        pass

    def directed_graph_weights(self,):
        pass

    def get_weights(self,):
        pass