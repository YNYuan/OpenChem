from openchem.models.atomic_conv import AtomicConvModel
from openchem.modules.mlp.openchem_mlp import OpenChemMLP
from openchem.data.graph_data_layer import GraphDataset_Atom
#from openchem.data.utils import get_dataset

import torch.nn as nn
from torch.optim import RMSprop, SGD, Adam
from torch.optim.lr_scheduler import ExponentialLR, StepLR
import torch.nn.functional as F
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

import copy
import pickle

# for the case we have dataset
def random_index(length, train_size):
    indices = np.random.RandomState(seed=32).permutation(length)
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

def get_dataset(xpath, zpath, nbrspath, nbrszpath, targetpath):
    train_idx, test_idx = random_index(2770, 0.8)

    trainX, testX = my_feature_split(xpath, train_idx, test_idx)
    trainZ, testZ = my_feature_split(zpath, train_idx, test_idx)
    trainNbrs, testNbrs = my_feature_split(nbrspath, train_idx, test_idx)
    trainNbrs_Z, testNbrs_Z = my_feature_split(nbrszpath, train_idx, test_idx)
    trainY, testY = my_target_split(targetpath, train_idx, test_idx)
    train_dataset = GraphDataset_Atom(trainX, trainZ, trainNbrs, trainNbrs_Z, trainY)
    test_dataset = GraphDataset_Atom(testX, testZ, testNbrs, testNbrs_Z, testY)
    return train_dataset, test_dataset

train_dataset, test_dataset = get_dataset('../3d_dataset/ligand_matrix.npy', '../3d_dataset/ligand_atomtype.npy', 
'../3d_dataset/ligand_distance_matrix.npy', '../3d_dataset/ligand_atomtype_matrix.npy', '../3d_dataset/whole_data.csv')

#for the case we don't have dataset, only have raw PDB data
#TODO

model = AtomicConvModel

model_params = {
    'task': 'regression',
    'data_layer': GraphDataset_Atom,
    'use_clip_grad': False,
    'batch_size': 12,
    'num_epochs': 100,
    'logdir': './acnnlogs_ligand',
    'print_every': 10,
    'save_every': 5,
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
