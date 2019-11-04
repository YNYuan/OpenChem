import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class AtomicConvolution(nn.Module):

    def __init__(self, atom_types=None, 
    radial_param=list(), boxsize=None):
        super(AtomicConvolution, self).__init__()
        self.boxsize = boxsize
        self.radial_params = radial_params
        self.atom_types = atom_types
        vars = []
        for i in range(3):
            val = np.array([p[i] for p in self.radial_params]).reshape((-1, 1, 1, 1))
            vars.append(Parameter(torch.FloatTensor(val)))
        self.rc = vars[0]
        self.rs = vars[1]
        self.re = vars[2]
    
    def forward(self, inputs):
        X = inputs[0]
        Nbrs = torch.IntTensor(inputs[1])
        Nbrs_Z = inputs[2]
        N = X.size()[-2]
        d = X.size()[-1]
        M = Nbrs.size()[-1]
        B = X.size()[0]
        D = self.distance_tensor(X, Nbrs, self.boxsize, B, N, M, d)
        R = self.distance_matrix(D)
        R = torch.unsqueeze(R, 0)
        rsf = self.radial_symmetry_function(R, self.rc, self.rs, self.re)
        if not self.atom_types:
            cond = torch.ne(Nbrs_Z, 0).float()
            cond = cond.view(1, -1, N, M)
            layer = torch.sum(cond * rsf, 3)
        else:
            sym = []
            for j in range(len(self.atom_types)):
                cond = torch.eq(Nbrs_Z, self.atom_types[j]).float()
                cond = cond.view(1, -1, N, M)
                sym.append(torch.sum(cond * rsf, 3))
            layer = torch.cat(sym, 0)

        layer = layer.permute(1, 2, 0)  # (l, B, N) -> (B, N, l)
        bn = nn.BatchNorm1d(layer.size()[1], track_running_stats=True)
        return bn(layer)

    def radial_symmetry_function(self, R, rc, rs, e):
        K = self.gaussian_distance_matrix(R, rs, e)
        FC = self.radial_cutoff(R, rc)
        return torch.mul(K, FC)

    def radial_cutoff(self, R, rc):
        T = 0.5 * (torch.cos(np.pi * R / (rc)) + 1)
        E = torch.zeros_like(T)
        cond = torch.le(R, rc)
        FC = torch.where(cond, T, E)
        return FC

    def gaussian_distance_matrix(self, R, rs, e):
        return torch.exp(-e * (R - rs)**2)

    def distance_tensor(self, X, Nbrs, boxsize, B, N, M, d):
        flat_neighbors = torch.reshape(Nbrs, [-1, N * M])
        neighbor_coords = torch.FloatTensor(B, N*M, d)
        for i in range(X.shape[0]):
            neighbor_coords[i] = torch.index_select(X[i], 0, flat_neighbors[i])
        neighbor_coords = neighbor_coords.view(-1, N, M, d)
        D = neighbor_coords - torch.unsqueeze(X, 2)
        if boxsize is not None:
            boxsize = boxsize.view(1, 1, 1, d)
            D -= torch.round(D / boxsize) * boxsize
        return D

    def distance_matrix(self, D):
        R = torch.sum(torch.mul(D, D), 3)
        R = torch.sqrt(R)
        return R