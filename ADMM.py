import numpy as np
import torch
import os
from utils import *


class ADMM_algorithm():
    '''
    only with 1 head
    '''
    def __init__(self, graph_info, ADMM_info, use_kNN=False, k=4, u_sigma=None, d_sigma=None, expand_time_dim=True, ablation='None', t_in=12, T=24):
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

        self.ablation = ablation
        assert ablation in ['None', 'DGTV', 'DGLR', 'UT'], "ablation should be in [\'None\', \'DGTV\', \'DGLR\', \'UT\']"
        self.u_ew = undirected_graph_from_distance(self.connect_list, self.dist_list, sigma=u_sigma, regularized=True)
        if self.ablation != 'UT':
            self.d_ew = undirected_graph_from_distance(self.connect_list, self.dist_list, sigma=u_sigma, regularized=True)

        if expand_time_dim:
            self.u_ew, self.d_ew = expand_time_dimension(self.u_ew, self.d_ew) # in (T, N, k)
        
        self.rho = ADMM_info['rho']
        self.rho_u = ADMM_info['rho_u']
        self.rho_d = ADMM_info['rho_d']
        self.mu_u = ADMM_info['mu_u']
        self.mu_d1 = ADMM_info['mu_d1']
        self.mu_d2 = ADMM_info['mu_d2']

        self.t_in = t_in
        self.T = T

        

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
    
    def apply_op_Ln(self, x): # undirected graphs
        pass
    
    def CG_solver(self, LHS_func, RHS, x0=None, tol=1e-10, max_iter=1000):
        '''
        Solving linear systems LHS_func(x) = RHS, B samples at the same time
        Input:
            x0 in (B, T, N, n_channels)
            LHS_func(x, args) in (B, T, N, n_channels)
            RHS in (B, T, N, n_channels)
        '''
        if x0 is None:
            x = torch.zeros_like(RHS)
        else:
            x = x0.clone()
        
        r = RHS - LHS_func(x)
        p = r.clone() # in (B, T, N, C)

        r_norm_sq = (r * r).sum((1,2,3)) # in (B,)
        for k in range(max_iter):
            Ap = LHS_func(p)
            alpha = r_norm_sq / (p * Ap).sum((1,2,3)) # in (B,)
            x = x + alpha[:, None, None, None] * p
            r = r - alpha[:, None, None, None] * Ap

            r_norm_new_sq = (r * r).sum((1,2,3)) # in (B,)
            beta = r_norm_new_sq / r_norm_sq
            r_norm_sq = r_norm_new_sq

            if torch.sqrt(r_norm_sq).max() < tol:
                return x, k + 1 # iterations
            
            p = r + beta[:, None, None, None] * p
        return x, max_iter


    def LHS_x(self, x):
        HtHx = x.clone()
        HtHx[:,self.t_in:] = HtHx[:,self.t_in:] * 0

        if self.ablation == 'None':
            output = HtHx + (self.rho_u + self.rho_d) / 2 * x + self.rho / 2 * self.apply_op_cLdr(x)
        elif self.ablation == 'DGTV':
            output = HtHx + (self.rho_u + self.rho_d) / 2 * x
        elif self.ablation == 'DGLR':
            output = HtHx + self.rho / 2 * self.apply_op_cLdr(x) + self.rho_u / 2 * x
        else: # 'UT'
            output = None # TODO: add undirected version
        return output

    def LHS_zu(self, zu):
        return self.mu_u * self.apply_op_Lu(zu) + self.rho_u / 2 * zu
    
    def LHS_zd(self, zd):
        return self.mu_d2 * self.apply_op_cLdr(zd) + self.rho_d / 2 * zd
    
    def phi_direct(self, x, gamma):
        '''
        phi^{tau+1} = soft_(mu_d1 / rho) (L^d_r x - gamma / rho)
        '''
        s = self.apply_op_Ldr(x) - gamma / self.rho
        d = self.mu_d1 / self.rho
        u = torch.abs(s) - d
        return torch.sign(s) * u * (u > 0)        

    def forward(self, y, mask=None, simple=False):
        '''
        Input:
            y in (B, t_in, N, C)
        Output:
            x in (B, T, N, C)
        actually the ADMMBlock accepts x
        '''
        # TODO: primal guess of x
        x = primal_guess(y, self.T)

        assert not torch.isnan(self.d_ew).any(), 'Directed graph weights d_ew has NaN value'
        assert not torch.isnan(self.u_ew).any(), 'Undirected graph weights u_ew has NaN value'
        
        gamma_u, gamma_d = torch.ones_like(x) * 0.05, torch.ones_like(x) * 0.1

        if self.ablation in ['None', 'DGLR']:
            gamma = torch.ones_like(x) * 0.1
            phi = self.apply_op_Ldr(x)
            assert not torch.isnan(phi).any(), 'initial phi has NaN value'
            
        zu, zd = x.clone(), x.clone()
        
        for i in range(self.ADMM_iters):
            # zu_old, zd_old = zu.clone(), zd.clone()
            # phi_old = phi.clone()
            Hty = torch.zeros_like(x)
            Hty[:,0:y.size(1)] = y
            if simple:
                RHS_x = self.apply_op_Ldr_T(self.rho * phi + gamma) / 2 + Hty
                assert not torch.isnan(RHS_x).any(), f'RHS_x has NaN value in loop {i}, d_ew in {self.d_ew.max().item():.4f}, {self.d_ew.min().item():.4f}'
                assert not torch.isinf(RHS_x).any() and not torch.isinf(-RHS_x).any(), f'RHS_x has inf value in loop {i}'
                x = self.CG_solver(self.LHS_simple_x, RHS_x, x, i, self.alpha_x, self.beta_x)
                assert not torch.isnan(x).any(), f'RHS_x has NaN value in loop {i}'
                assert not torch.isinf(x).any() and not torch.isinf(x).any(), f'x has inf value in loop {i}'
            else:
                # print(torch.isnan(gamma + self.rho[i] * phi).any(), torch.isnan(gamma).any(), )
                if self.ablation == 'DGTV':
                    RHS_x = (self.rho_u[i] * zu + self.rho_d[i] * zd) / 2 - (gamma_u + gamma_d) / 2 + Hty
                elif self.ablation == 'None':
                    RHS_x = self.apply_op_Ldr_T(gamma + self.rho[i] * phi) / 2 + (self.rho_u[i] * zu + self.rho_d[i] * zd) / 2 - (gamma_u + gamma_d) / 2 + Hty
                elif self.ablation == 'UT':
                    pass
                elif self.ablation == 'DGLR':
                    RHS_x = self.apply_op_Ldr_T(gamma + self.rho[i] * phi) / 2 + self.rho_u[i] * zu / 2 - gamma_u / 2 + Hty
                # print(torch.isnan(zu).any(), torch.isnan(zd).any())
                    assert not torch.isnan(gamma + self.rho[i] * phi).any(), f'Ldr T input has NaN in loop {i}, gamma {torch.isnan(gamma).any()}, rho[i] {torch.isnan(self.rho[i]).any()}, phi {torch.isnan(phi).any()}'

                assert not torch.isnan(RHS_x).any(), f'RHS_x has NaN value in loop {i}, d_ew in {self.d_ew.max().item():.4f}, {self.d_ew.min().item():.4f}, NaN {torch.isnan(self.d_ew).any()}, (rho_u, rho_d)[i] has NaN ({torch.isnan(self.rho_u[i]).any()}, {torch.isnan(self.rho_d[i]).any()}), (z_u, z_d) has NaN ({torch.isnan(zu).any()}, {torch.isnan(zd).any()}), (gamma_u, gamma_d) has NaN ({torch.isnan(gamma_u).any()}, {torch.isnan(gamma_d).any()})'
                assert not torch.isinf(RHS_x).any() and not torch.isinf(-RHS_x).any(), f'RHS_x has inf value in loop {i}'
                # print('RHS_x', torch.isnan(RHS_x).any(), RHS_x.max(), RHS_x.min())
                # solve x with zu, zd, update x
                x = self.CG_solver(self.LHS_x, RHS_x, x, i, self.alpha_x, self.beta_x, args=y)
                assert not torch.isnan(x).any(), f'RHS_x has NaN value in loop {i}'
                assert not torch.isinf(x).any() and not torch.isinf(x).any(), f'x has inf value in loop {i}'
                # print('x', torch.isnan(x).any())
                # solve zu, zd with x, update zu, zd
                RHS_zu = gamma_u / 2 + self.rho_u[i] / 2 * x
                zu = self.CG_solver(self.LHS_zu, RHS_zu, zu, i, self.alpha_zu, self.beta_zu)
                assert not torch.isnan(RHS_zu).any(), f'RHS_zu has NaN value in loop {i}'
                assert not torch.isinf(RHS_zu).any() and not torch.isinf(-RHS_zu).any(), f'RHS_zu has inf value in loop {i}'
                # print('RHS_zu, zu', torch.isnan(RHS_zu).any(), RHS_zu.max(), RHS_zu.min(), torch.isnan(zu).any(), zu.max(), zu.min())
                if self.ablation != 'DGLR':
                    RHS_zd = gamma_d / 2 + self.rho_d[i] / 2 * x
                    zd = self.CG_solver(self.LHS_zd, RHS_zd, zd, i, self.alpha_zd, self.beta_zd)
                    assert not torch.isnan(RHS_zd).any(), f'RHS_zd has NaN value in loop {i}'
                    assert not torch.isinf(RHS_zd).any() and not torch.isinf(-RHS_zd).any(), f'RHS_zd has inf value in loop {i}'

                gamma_u = gamma_u + self.rho_u[i] * (x - zu)
                if self.ablation != 'DGLR':
                    gamma_d = gamma_d + self.rho_d[i] * (x - zd)
            # udpata phi
            # phi = self.Phi_PGD(phi, x, gamma, i) # 
            if self.ablation in ['None', 'DGLR']:
                # print('executed phi update')
                phi = self.phi_direct(x, gamma, i)
                gamma = gamma + self.rho[i] * (phi - self.apply_op_Ldr(x))
                assert not (torch.isnan(gamma).any() or torch.isnan(phi).any()), f'gamma has NaN {torch.isnan(gamma).any()}, phi has NaN {torch.isnan(phi).any()}'
        #     # criterion
        #     primal_residual = max(torch.norm(phi - self.apply_op_Ldr(x)), torch.norm(x - zu), torch.norm(x - zd))
        #     dual_residual = max(torch.norm(-self.rho[i] * self.apply_op_Ldr_T(phi - phi_old)), torch.norm(-self.rho_u[i] * (zu - zu_old)), torch.norm(-self.rho_d[i] * (zd - zd_old)))
        #     print(f'ADMM iters {i}: pri_err = {primal_residual}, dual_err = {dual_residual}')
        #     if primal_residual < 1e-3 and dual_residual < 1e-3:
        #         break
        # print(f'single ADMM iters {i}')
        # combination weights
        #output = self.comb_fc(x.transpose(-2, -1)).squeeze(-1)
        output = torch.einsum('btnhc, h -> btnc', x, self.comb_weights)
        # print('ADMM out', output.max(), output.min())
        return output

    def forward(self,):
        pass

def primal_guess(y, T):
    '''
    y in (B, t_in, N, C)
    return: x in (B, T, N, C)
    use a simple linear regression to guess x
    '''


if __name__ == '__main__':
    # data_dir
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

# graph construction