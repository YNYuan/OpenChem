from openchem.models.atomic_conv import AtomicConvModel0, AtomicConvModel
from openchem.modules.mlp.openchem_mlp import OpenChemMLP
from openchem.data.graph_data_layer import GraphDataset_Single, GraphDataset_Multi
from openchem.data.ac_data_reader import ACDataReader

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

myReader = ACDataReader(2770, 0.8)
# train_dataset_l, test_dataset_l = myReader.get_dataset_single('../3d_dataset/ligand_coord.npy', '../3d_dataset/ligand_type.npy', 
# '../3d_dataset/ligand_distance_matrix.npy', '../3d_dataset/ligand_atomtype_matrix.npy', '../3d_dataset/whole_data.csv')
train_dataset, test_dataset = myReader.get_dataset_multi('../3d_dataset/ligand_coord.npy', 
'../3d_dataset/ligand_type.npy', '../3d_dataset/ligand_distance_matrix.npy', 
'../3d_dataset/ligand_atomtype_matrix.npy', '../3d_dataset/rec_coord_new.npy', 
'../3d_dataset/rec_type_new.npy', '../3d_dataset/rec_distance_matrix.npy', 
'../3d_dataset/rec_atomtype_matrix.npy', '../3d_dataset/complex_coord_new.npy', 
'../3d_dataset/complex_type_new.npy', '../3d_dataset/complex_distance_matrix.npy', 
'../3d_dataset/complex_atomtype_matrix.npy', '../3d_dataset/whole_data.csv')

#for the case we don't have dataset, only have raw PDB data
#run 'data_generator.py' first, then run this script

model = AtomicConvModel

model_params = {
    'task': 'regression',
    'data_layer': GraphDataset_Multi,
    'use_clip_grad': False,
    'batch_size': 24,
    'num_epochs': 100,
    'logdir': './acnnlogs',
    'print_every': 10,
    'save_every': 4,
    'train_data_layer': train_dataset,
    'val_data_layer': test_dataset,
    'eval_metrics': r2_score,
    'criterion': nn.MSELoss(),
    'optimizer': Adam,
    'optimizer_params': {
        'lr': 0.01,
    },
    'lr_scheduler': StepLR,
    'lr_scheduler_params': {
        'step_size': 50,
        'gamma': 0.9
    },
    'mlp': OpenChemMLP,
    'mlp_params': {
        'input_size': 594,
        'n_layers': 3,
        'hidden_size': [8, 4, 1],
        'activation': [F.relu, F.relu, nn.Identity()],
    },
    'radial': [[
        1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0,
        7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0], [0.0, 4.0, 8.0], [0.4]],
    'atom_types': [
        6., 7., 8., 9., 15., 16., 17., 35., 53.],
    'random_seed': 42
}
