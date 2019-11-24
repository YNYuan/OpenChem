from openchem.models.atomic_conv import AtomicConvModel0
from openchem.modules.mlp.openchem_mlp import OpenChemMLP

import torch.nn as nn
from torch.optim import RMSprop, SGD, Adam
from torch.optim.lr_scheduler import ExponentialLR, StepLR
from torch.utils.data import Dataset
import torch.nn.functional as F
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

import copy
import pickle

def random_index(length, train_size):
    indices = np.random.permutation(length)
    cutNum = int(np.floor(length*train_size))
    train_idx, test_idx = indices[:cutNum], indices[cutNum:]
    return train_idx, test_idx


def my_feature_split(matrix_path, train_idx, test_idx):
    matrix = np.load(matrix_path)
    train_matrix, test_matrix = matrix[train_idx,:,:], matrix[test_idx,:,:]
    return train_matrix, test_matrix


def my_target_split(whole_path, train_idx, test_idx):
    name_file = pd.read_csv(whole_path)
    target = name_file['affinity'].values
    trainY = target[train_idx].reshape(-1,1)
    testY = target[test_idx].reshape(-1,1)
    return trainY, testY


class GraphDataset(Dataset):
    def __init__(self, X_matrix_c, Z_matrix_c, R_matrix_c, Nbrs_Z_matrix_c, X_matrix_l, Z_matrix_l, R_matrix_l, Nbrs_Z_matrix_l, X_matrix_r, Z_matrix_r, R_matrix_r, Nbrs_Z_matrix_r, target):
        super(GraphDataset, self).__init__()
        self.X_matrix_c = X_matrix_c
        self.Z_matrix_c = Z_matrix_c
        #self.Nbrs_matrix = Nbrs_matrix
        self.R_matrix_c = R_matrix_c
        self.Nbrs_Z_matrix_c = Nbrs_Z_matrix_c
        self.X_matrix_l = X_matrix_l
        self.Z_matrix_l = Z_matrix_l
        self.R_matrix_l = R_matrix_l
        self.Nbrs_Z_matrix_l = Nbrs_Z_matrix_l
        self.X_matrix_r = X_matrix_r
        self.Z_matrix_r = Z_matrix_r
        self.R_matrix_r = R_matrix_r
        self.Nbrs_Z_matrix_r = Nbrs_Z_matrix_r
        self.target = target
        #self.num_features = X_matrix.shape[2]

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        sample = {
            'X_matrix_c': self.X_matrix_c[index].astype('float32'),
            'Z_matrix_c': self.Z_matrix_c[index].astype('float32'),
            # 'Nbrs_matrix': self.Nbrs_matrix[index].astype('float32'),
            'R_matrix_c': self.R_matrix_c[index].astype('float32'),
            'Nbrs_Z_matrix_c': self.Nbrs_Z_matrix_c[index].astype('float32'),
            'X_matrix_l': self.X_matrix_l[index].astype('float32'),
            'Z_matrix_l': self.Z_matrix_l[index].astype('float32'),
            'R_matrix_l': self.R_matrix_l[index].astype('float32'),
            'Nbrs_Z_matrix_l': self.Nbrs_Z_matrix_l[index].astype('float32'),
            'X_matrix_r': self.X_matrix_r[index].astype('float32'),
            'Z_matrix_r': self.Z_matrix_r[index].astype('float32'),
            'R_matrix_r': self.R_matrix_r[index].astype('float32'),
            'Nbrs_Z_matrix_r': self.Nbrs_Z_matrix_r[index].astype('float32'),
            'target': self.target[index].astype('float32')
        }
        return sample

def get_dataset(xpath_c, zpath_c, nbrspath_c, nbrszpath_c, xpath_l, zpath_l, nbrspath_l, nbrszpath_l, xpath_r, zpath_r, nbrspath_r, nbrszpath_r, targetpath):
    train_idx, test_idx = random_index(277, 0.8)

    trainX_c, testX_c = my_feature_split(xpath_c, train_idx, test_idx)
    trainZ_c, testZ_c = my_feature_split(zpath_c, train_idx, test_idx)
    trainNbrs_c, testNbrs_c = my_feature_split(nbrspath_c, train_idx, test_idx)
    trainNbrs_Z_c, testNbrs_Z_c = my_feature_split(nbrszpath_c, train_idx, test_idx)
    trainX_l, testX_l = my_feature_split(xpath_l, train_idx, test_idx)
    trainZ_l, testZ_l = my_feature_split(zpath_l, train_idx, test_idx)
    trainNbrs_l, testNbrs_l = my_feature_split(nbrspath_l, train_idx, test_idx)
    trainNbrs_Z_l, testNbrs_Z_l = my_feature_split(nbrszpath_l, train_idx, test_idx)
    trainX_r, testX_r = my_feature_split(xpath_r, train_idx, test_idx)
    trainZ_r, testZ_r = my_feature_split(zpath_r, train_idx, test_idx)
    trainNbrs_r, testNbrs_r = my_feature_split(nbrspath_c, train_idx, test_idx)
    trainNbrs_Z_r, testNbrs_Z_r = my_feature_split(nbrszpath_r, train_idx, test_idx)
    trainY, testY = my_target_split(targetpath, train_idx, test_idx)
    train_dataset = GraphDataset(trainX_c, trainZ_c, trainNbrs_c, trainNbrs_Z_c, trainX_l, trainZ_l, trainNbrs_l, trainNbrs_Z_l, trainX_r, trainZ_r, trainNbrs_r, trainNbrs_Z_r, trainY)
    test_dataset = GraphDataset(testX_c, testZ_c, testNbrs_c, testNbrs_Z_c, testX_l, testZ_l, testNbrs_l, testNbrs_Z_l, testX_r, testZ_r, testNbrs_r, testNbrs_Z_r, testY)
    return train_dataset, test_dataset

train_dataset, test_dataset = get_dataset('../3d_dataset/complex_matrix.npy', '../3d_dataset/complex_type.npy', 
'../3d_dataset/complex_distance_matrix.npy', '../3d_dataset/complex_atomtype_matrix.npy', '../3d_dataset/ligand_matrix.npy', '../3d_dataset/ligand_atomtype.npy', 
'../3d_dataset/ligand_distance_matrix.npy', '../3d_dataset/ligand_atomtype_matrix.npy', '../3d_dataset/rec_matrix.npy', '../3d_dataset/rec_atomtype.npy', 
'../3d_dataset/rec_distance_matrix.npy', '../3d_dataset/rec_atomtype_matrix.npy', '../3d_dataset/whole_data.csv')

model = AtomicConvModel0

model_params = {
    'task': 'regression',
    'data_layer': GraphDataset,
    'use_clip_grad': False,
    'batch_size': 2,
    'num_epochs': 10,
    'logdir': './acnnlogs',
    'print_every': 1,
    'save_every': 1,
    'train_data_layer': train_dataset,
    'val_data_layer': test_dataset,
    'eval_metrics': r2_score,
    'criterion': nn.MSELoss(),
    'optimizer': Adam,
    'optimizer_params': {
        'lr': 0.001,
    },
    'lr_scheduler': StepLR,
    'lr_scheduler_params': {
        'step_size': 15,
        'gamma': 0.5
    },
    'mlp': OpenChemMLP,
    'mlp_params': {
        'input_size': 594,
        'n_layers': 4,
        'hidden_size': [32, 32, 16, 1],
        'activation': [F.relu, F.relu, F.relu, nn.Identity()],
    },
    'radial': [[
        1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0,
        7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0], [0.0, 4.0, 8.0], [0.4]],
    'atom_types': [
        6, 7., 8., 9., 15., 16., 17., 35., 53.],
    'random_seed': 42
}
