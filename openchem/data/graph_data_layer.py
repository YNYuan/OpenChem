# TODO: variable length batching

import numpy as np


from openchem.utils.graph import Graph

from rdkit import Chem

from torch.utils.data import Dataset
from openchem.data.utils import read_smiles_property_file, sanitize_smiles


class GraphDataset(Dataset):
    def __init__(self, get_atomic_attributes, node_attributes, filename,
                 cols_to_read, delimiter=',', get_bond_attributes=None, edge_attributes=None):
        super(GraphDataset, self).__init__()
        assert (get_bond_attributes is None) == (edge_attributes is None)
        data_set = read_smiles_property_file(filename, cols_to_read,
                                             delimiter)
        data = data_set[0]
        target = data_set[1:]
        clean_smiles, clean_idx = sanitize_smiles(data)
        target = np.array(target).T
        max_size = 0
        for sm in clean_smiles:
            mol = Chem.MolFromSmiles(sm)
            if mol.GetNumAtoms() > max_size:
                max_size = mol.GetNumAtoms()
        self.target = target[clean_idx, :]
        self.graphs = []
        self.node_feature_matrix = []
        self.adj_matrix = []
        for sm in clean_smiles:
            graph = Graph(sm, max_size, get_atomic_attributes,
                          get_bond_attributes)
            self.node_feature_matrix.append(
                graph.get_node_feature_matrix(node_attributes, max_size))
            if get_bond_attributes is None:
                self.adj_matrix.append(graph.adj_matrix)
            else:
                self.adj_matrix.append(
                    graph.get_edge_attr_adj_matrix(edge_attributes, max_size)
                )
        self.num_features = self.node_feature_matrix[0].shape[1]

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        sample = {'adj_matrix': self.adj_matrix[index].astype('float32'),
                  'node_feature_matrix':
                      self.node_feature_matrix[index].astype('float32'),
                  'labels': self.target[index].astype('float32')}
        return sample


class GraphDataset_Single(Dataset):
    def __init__(self, X_matrix, Z_matrix, R_matrix, Nbrs_Z_matrix, target):
        super(GraphDataset_Atom, self).__init__()
        self.X_matrix = X_matrix
        self.Z_matrix = Z_matrix
        #self.Nbrs_matrix = Nbrs_matrix
        self.R_matrix = R_matrix
        self.Nbrs_Z_matrix = Nbrs_Z_matrix
        self.target = target

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        sample = {
            'X_matrix': self.X_matrix[index].astype('float32'),
            'Z_matrix': self.Z_matrix[index].astype('float32'),
            # 'Nbrs_matrix': self.Nbrs_matrix[index].astype('float32'),
            'R_matrix': self.R_matrix[index].astype('float32'),
            'Nbrs_Z_matrix': self.Nbrs_Z_matrix[index].astype('float32'),
            'target': self.target[index].astype('float32')
        }
        return sample


class GraphDataset_Multi(Dataset):
    def __init__(self, X_matrix_l, Z_matrix_l, R_matrix_l, Nbrs_Z_matrix_l, X_matrix_r, 
    Z_matrix_r, R_matrix_r, Nbrs_Z_matrix_r, X_matrix_c, Z_matrix_c, R_matrix_c, 
    Nbrs_Z_matrix_c, target):
        super(GraphDataset_Atom, self).__init__()
        self.X_matrix_l = X_matrix_l
        self.Z_matrix_l = Z_matrix_l
        #self.Nbrs_matrix = Nbrs_matrix
        self.R_matrix_l = R_matrix_l
        self.Nbrs_Z_matrix_l = Nbrs_Z_matrix_l
        self.X_matrix_r = X_matrix_r
        self.Z_matrix_r = Z_matrix_r
        self.R_matrix_r = R_matrix_r
        self.Nbrs_Z_matrix_r = Nbrs_Z_matrix_r
        self.X_matrix_c = X_matrix_c
        self.Z_matrix_c = Z_matrix_c
        self.R_matrix_c = R_matrix_c
        self.Nbrs_Z_matrix_c = Nbrs_Z_matrix_c
        self.target = target

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        sample = {
            'X_matrix_l': self.X_matrix_l[index].astype('float32'),
            'Z_matrix_l': self.Z_matrix_l[index].astype('float32'),
            # 'Nbrs_matrix': self.Nbrs_matrix[index].astype('float32'),
            'R_matrix_l': self.R_matrix_l[index].astype('float32'),
            'Nbrs_Z_matrix_l': self.Nbrs_Z_matrix_l[index].astype('float32'),
            'X_matrix_r': self.X_matrix_r[index].astype('float32'),
            'Z_matrix_r': self.Z_matrix_r[index].astype('float32'),
            'R_matrix_r': self.R_matrix_r[index].astype('float32'),
            'Nbrs_Z_matrix_r': self.Nbrs_Z_matrix_r[index].astype('float32'),
            'X_matrix_c': self.X_matrix_c[index].astype('float32'),
            'Z_matrix_c': self.Z_matrix_c[index].astype('float32'),
            'R_matrix_c': self.R_matrix_c[index].astype('float32'),
            'Nbrs_Z_matrix_c': self.Nbrs_Z_matrix_c[index].astype('float32'),
            'target': self.target[index].astype('float32')
        }
        return sample