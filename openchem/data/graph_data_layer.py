# TODO: variable length batching

import torch
import numpy as np
import networkx as nx
import pickle

from openchem.utils.graph import Graph

from torch.utils.data import Dataset
from openchem.data.utils import read_smiles_property_file, sanitize_smiles

from .graph_utils import bfs_seq, encode_adj


class GraphDataset(Dataset):
    def __init__(self, get_atomic_attributes, node_attributes, filename,
                 cols_to_read, delimiter=',', get_bond_attributes=None,
                 edge_attributes=None,
                 restrict_min_atoms=-1, restrict_max_atoms=-1,
                 **kwargs):
        super(GraphDataset, self).__init__()
        assert (get_bond_attributes is None) == (edge_attributes is None)

        if "pickled" in kwargs:
            data = pickle.load(open(kwargs["pickled"], "rb"))

            self.num_atoms_all = data["num_atoms_all"]
            self.target = data["target"]
            self.smiles = data["smiles"]
        else:
            data_set = read_smiles_property_file(filename, cols_to_read,
                                                 delimiter)
            data = data_set[0]
            target = data_set[1:]
            clean_smiles, clean_idx, num_atoms = sanitize_smiles(
                data,
                min_atoms=restrict_min_atoms,
                max_atoms=restrict_max_atoms,
                return_num_atoms=True
            )
            target = np.asarray(target, dtype=np.float).T

            self.target = target[clean_idx, :]
            self.smiles = clean_smiles
            self.num_atoms_all = num_atoms

        self.max_size = max(self.num_atoms_all)

        self.node_attributes = node_attributes
        self.edge_attributes = edge_attributes
        self.get_atomic_attributes = get_atomic_attributes
        self.get_bond_attributes = get_bond_attributes

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        sm = self.smiles[index]

        graph = Graph(
            sm, self.max_size, self.get_atomic_attributes,
            self.get_bond_attributes)
        node_feature_matrix = graph.get_node_feature_matrix(
            self.node_attributes, self.max_size)

        # TODO: remove diagonal elements from adjacency matrix
        if self.get_bond_attributes is None:
            adj_matrix = graph.adj_matrix
        else:
            adj_matrix = graph.get_edge_attr_adj_matrix(
                self.edge_attributes, self.max_size)

        sample = {'adj_matrix': adj_matrix.astype('float32'),
                  'node_feature_matrix':
                      node_feature_matrix.astype('float32'),
                  'labels': self.target[index].astype('float32')}
        return sample


class BFSGraphDataset(GraphDataset):
    def __init__(self, *args, **kwargs):
        super(BFSGraphDataset, self).__init__(*args, **kwargs)
        self.random_order = kwargs["random_order"]
        self.max_prev_nodes = kwargs["max_prev_nodes"]
        self.num_edge_classes = kwargs
        self.max_num_nodes = max(self.num_atoms_all)
        original_start_node_label = kwargs.get(
            "original_start_node_label", None)

        if "node_relabel_map" not in kwargs:
            # define relabelling from Periodic Table numbers to {0, 1, ...}
            unique_labels = set()
            for index in range(len(self)):
                sample = super(BFSGraphDataset, self).__getitem__(index)
                node_feature_matrix = sample['node_feature_matrix']
                adj_matrix = sample['adj_matrix']

                labels = set(node_feature_matrix.flatten().tolist())
                unique_labels.update(labels)

            # discard 0 padding
            unique_labels.discard(0)

            self.node_relabel_map = {
                v: i for i, v in enumerate(sorted(unique_labels))
            }
        else:
            self.node_relabel_map = kwargs["node_relabel_map"]
        self.inverse_node_relabel_map = {i: v for v, i in
                                         self.node_relabel_map.items()}

        if original_start_node_label is not None:
            self.start_node_label = \
                self.node_relabel_map[original_start_node_label]
        else:
            self.start_node_label = None

        if "edge_relabel_map" not in kwargs:
            raise NotImplementedError()
        else:
            self.edge_relabel_map = kwargs["edge_relabel_map"]
        self.inverse_edge_relabel_map = {
            i: v for v, i in
            sorted(self.edge_relabel_map.items(), reverse=True)}

        self.num_node_classes = len(self.inverse_node_relabel_map)
        self.num_edge_classes = len(self.inverse_edge_relabel_map)

    def __getitem__(self, index):
        sample = super(BFSGraphDataset, self).__getitem__(index)
        adj_original = sample['adj_matrix']
        node_feature_matrix = sample['node_feature_matrix']
        num_nodes = self.num_atoms_all[index]

        adj_original = adj_original.reshape(adj_original.shape[:2])
        adj = np.zeros_like(adj_original)
        for v, i in self.edge_relabel_map.items():
            adj[adj_original == v] = i

        labels = np.array(
            [self.node_relabel_map[v] if v != 0 else 0
             for v in node_feature_matrix.flatten()])

        if self.random_order:
            order = np.random.permutation(num_nodes)
            adj = adj[np.ix_(order, order)]
            labels = labels[order]

        adj_matrix = np.asmatrix(adj)
        G = nx.from_numpy_matrix(adj_matrix)

        if self.start_node_label is None:
            start_idx = np.random.randint(num_nodes)
        else:
            start_idx = np.random.choice(
                np.where(labels == self.start_node_label)[0])

        # BFS ordering
        order = np.array(bfs_seq(G, start_idx))
        adj = adj[np.ix_(order, order)]
        labels = labels[order]

        ii, jj = np.where(adj)
        max_prev_nodes_local = np.abs(ii - jj).max()

        # TODO: is copy needed here?
        adj_encoded = encode_adj(adj.copy(), max_prev_node=self.max_prev_nodes)

        adj_encoded = torch.tensor(adj_encoded, dtype=torch.float)
        labels = torch.tensor(labels, dtype=torch.long)

        x = torch.zeros((self.max_num_nodes, self.max_prev_nodes),
                        dtype=torch.float)
        # TODO: the first input token is all ones?
        x[0, :] = 1.
        y = torch.zeros((self.max_num_nodes, self.max_prev_nodes),
                        dtype=torch.long)
        c_in = torch.zeros(self.max_num_nodes, dtype=torch.long)
        c_out = -1 * torch.ones(self.max_num_nodes, dtype=torch.long)

        y[:num_nodes-1, :] = adj_encoded.to(dtype=torch.long)
        x[1:num_nodes, :] = adj_encoded
        c_in[:num_nodes] = labels
        c_out[:num_nodes-1] = labels[1:]

        return {'x': x, 'y': y, 'num_nodes': num_nodes,
                'c_in': c_in, 'c_out': c_out,
                'max_prev_nodes_local': max_prev_nodes_local}
