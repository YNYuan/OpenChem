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
    def __init__(self, X_matrix, Z_matrix, R_matrix, Nbrs_Z_matrix, target):
        super(GraphDataset, self).__init__()
        self.X_matrix = X_matrix
        self.Z_matrix = Z_matrix
        #self.Nbrs_matrix = Nbrs_matrix
        self.R_matrix = R_matrix
        self.Nbrs_Z_matrix = Nbrs_Z_matrix
        self.target = target
        self.num_features = X_matrix.shape[2]

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        sample = {'X_matrix': self.X_matrix[index].astype('float32'),
            'Z_matrix': self.Z_matrix[index].astype('float32'),
            # 'Nbrs_matrix':
            # self.Nbrs_matrix[index].astype('float32'),
            'R_matrix':
            self.R_matrix[index].astype('float32'),
            'Nbrs_Z_matrix':
            self.Nbrs_Z_matrix[index].astype('float32'),
            'target': self.target[index].astype('float32')
        }
        return sample

def get_dataset():
    train_idx, test_idx = random_index(2770, 0.8)

    trainX_complex, testX_complex = my_feature_split('../3d_dataset/complex_matrix.npy', train_idx, test_idx)
    trainZ_complex, testZ_complex = my_feature_split('../3d_dataset/complex_type.npy', train_idx, test_idx)
    trainNbrs_complex, testNbrs_complex = my_feature_split('../3d_dataset/complex_distance_matrix.npy', train_idx, test_idx)
    trainNbrs_Z_complex, testNbrs_Z_complex = my_feature_split('../3d_dataset/complex_atomtype_matrix.npy', train_idx, test_idx)
    trainY, testY = my_target_split('../3d_dataset/whole_data.csv', train_idx, test_idx)
    train_dataset = GraphDataset(trainX_complex, trainZ_complex, trainNbrs_complex, trainNbrs_Z_complex, trainY)
    test_dataset = GraphDataset(testX_complex, testZ_complex, testNbrs_complex, testNbrs_Z_complex, testY)
    return train_dataset, test_dataset

train_dataset, test_dataset = get_dataset()
model = AtomicConvModel0

model_params = {
    'task': 'regression',
    'data_layer': GraphDataset,
    'use_clip_grad': False,
    'batch_size': 6,
    'num_epochs': 100,
    'logdir': './acnnlogs',
    'print_every': 10,
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
        'input_size': 660,
        'n_layers': 4,
        'hidden_size': [32, 32, 16, 1],
        'activation': [F.relu, F.relu, F.relu, nn.Identity()],
    },
    'radial': [[
        1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0,
        7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0], [0.0, 4.0, 8.0], [0.4]],
    'atom_types': [
        6, 7., 8., 9., 15., 16., 17., 35., 53., 0.],
    'random_seed': 42
}
