from __future__ import division
from __future__ import unicode_literals

import sys
import numpy as np
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from openchem.layers.ac import AtomicConvolution
from openchem.models.openchem_model import OpenChemModel


def initializeWeightsBiases(prev_layer_size,
                            size,
                            weights=None,
                            biases=None,
                            name=None):

    if weights is None:
        weights = nn.init.normal_(torch.Tensor(prev_layer_size, size), mean=0, std=0.01)
    if biases is None:
        biases = torch.zeros([size])

    w = Parameter(weights)
    b = Parameter(biases)
    return w, b


class AtomicConvModel0(OpenChemModel):

  def __init__(self, params):
    super(AtomicConvModel0, self).__init__(params)

    self.radial = params['radial']
    self.atom_types = params['atom_types']
    self.MLP_list = nn.ModuleList([])
    for ind, atomtype in enumerate(self.atom_types):
      self.mlp = self.params['mlp'].cuda()
      self.mlp_params = self.params['mlp_params']
      self.MLP = self.mlp(self.mlp_params)
      self.MLP_list.append(self.MLP)

    rp = [x for x in itertools.product(*self.radial)]

    conv = AtomicConvolution(
        atom_types=self.atom_types, radial_params=rp,
        boxsize=None)
    self.conv = conv

  def forward(self, input, eval=False):
    if eval:
      self.eval()
    else:
      self.train()

    complex_X = input[0] 
    complex_Z = input[1]
    #complex_Nbrs = input[2]
    complex_R = input[2]
    complex_Nbrs_Z = input[3]
    
    complex_conv = self.conv.forward([complex_X, complex_R, complex_Nbrs_Z])
    complex_zeros = torch.zeros_like(complex_Z)
    complex_atomtype_energy = []
    for ind, atomtype in enumerate(self.atom_types):
      complex_outputs = torch.FloatTensor(complex_conv.shape[0], complex_conv.shape[1], 1)
      for i, sample in enumerate(complex_conv):
        complex_outputs[i] = self.MLP_list[ind](sample)
      cond = torch.eq(complex_Z, atomtype)
      complex_atomtype_energy.append(
        torch.where(cond, complex_outputs, complex_zeros))
    complex_outputs = torch.stack(complex_atomtype_energy, dim=0).sum(dim=0)
    complex_energy = torch.sum(complex_outputs, 1)
    return complex_energy#torch.unsqueeze(complex_energy, 1)

  def cast_inputs(self, sample):
    batch_X = sample['X_matrix'].clone().detach().requires_grad_(True)
    batch_Z = sample['Z_matrix'].clone().detach().requires_grad_(True)
    # batch_Nbrs = torch.tensor(sample['Nbrs_matrix'],
    #     requires_grad=True).float()
    batch_R = sample['R_matrix'].clone().detach().requires_grad_(True)
    batch_Nbrs_Z = sample['Nbrs_Z_matrix'].clone().detach().requires_grad_(True)
    batch_labels = sample['target'].clone().detach().requires_grad_(False)
    if self.task == 'classification':
        batch_labels = batch_labels.long()
    else:
        batch_labels = batch_labels.float()
    if self.use_cuda:
        batch_X = batch_X.cuda()
        batch_Z = batch_Z.cuda()
        #batch_Nbrs = batch_Nbrs.cuda()
        batch_R = batch_R.cuda()
        batch_Nbrs_Z = batch_Nbrs_Z.cuda()
        batch_labels = batch_labels.cuda()
    batch_inp = (batch_X, batch_Z, batch_R, batch_Nbrs_Z)
    return batch_inp, batch_labels


class AtomicConvScore(nn.Module):

  def __init__(self, atom_types, layer_sizes, input_shape):
      super(AtomicConvScore, self).__init__()
      self.atom_types = atom_types

      self.type_weights = []
      self.type_biases = []
      self.output_weights = []
      self.output_biases = []
      n_features = int(input_shape[0][-1])
      num_layers = len(layer_sizes)
      weight_init_stddevs = [1 / np.sqrt(x) for x in layer_sizes]
      bias_init_consts = [0.0] * num_layers
      for ind, atomtype in enumerate(self.atom_types):
          prev_layer_size = n_features
          self.type_weights.append([])
          self.type_biases.append([])
          self.output_weights.append([])
          self.output_biases.append([])
          for i in range(num_layers):
              weight, bias = initializeWeightsBiases(
                  prev_layer_size=prev_layer_size,
                  size=layer_sizes[i],
                  weights=nn.init.normal_(
                  torch.Tensor(prev_layer_size, layer_sizes[i]),
                  std=weight_init_stddevs[i]),
                  biases=nn.init.constant_(torch.Tensor(layer_sizes[i]),
                  val=bias_init_consts[i]))
              self.type_weights[ind].append(weight)
              self.type_biases[ind].append(bias)
              prev_layer_size = layer_sizes[i]
          weight, bias = initializeWeightsBiases(prev_layer_size, 1)
          self.output_weights[ind].append(weight)
          self.output_biases[ind].append(bias)

  def forward(self, inputs):
      frag1_layer, frag2_layer, complex_layer, frag1_z, frag2_z, complex_z = inputs
      atom_types = self.atom_types
      num_layers = len(self.layer_sizes)

      def atomnet(current_input, atomtype):
          prev_layer = current_input
          for i in range(num_layers):
              layer = F.linear(prev_layer, weight=self.type_weights[atomtype][i],
                              bias=self.type_biases[atomtype][i])
              layer = F.relu(layer)
              prev_layer = layer

          output_layer = torch.squeeze(
              F.linear(prev_layer, weight=self.output_weights[atomtype][0],
                        bias=self.output_biases[atomtype][0]))
          return output_layer

      frag1_zeros = torch.zeros_like(frag1_z)
      frag2_zeros = torch.zeros_like(frag2_z)
      complex_zeros = torch.zeros_like(complex_z)

      frag1_atomtype_energy = []
      frag2_atomtype_energy = []
      complex_atomtype_energy = []

      for ind, atomtype in enumerate(atom_types):
          frag1_outputs = atomnet(frag1_layer, ind)
          frag2_outputs = atomnet(frag2_layer, ind)
          complex_outputs = atomnet(complex_layer, ind)

          cond = torch.eq(frag1_z, atomtype)
          frag1_atomtype_energy.append(torch.where(cond, frag1_outputs, frag1_zeros))
          cond = torch.eq(frag2_z, atomtype)
          frag2_atomtype_energy.append(torch.where(cond, frag2_outputs, frag2_zeros))
          cond = torch.eq(complex_z, atomtype)
          complex_atomtype_energy.append(
              torch.where(cond, complex_outputs, complex_zeros))

      frag1_outputs = torch.stack(frag1_atomtype_energy, dim=0).sum(dim=0)
      frag2_outputs = torch.stack(frag2_atomtype_energy, dim=0).sum(dim=0)
      complex_outputs = torch.stack(complex_atomtype_energy, dim=0).sum(dim=0)

      frag1_energy = torch.sum(frag1_outputs, 1)
      frag2_energy = torch.sum(frag2_outputs, 1)
      complex_energy = torch.sum(complex_outputs, 1)
      binding_energy = complex_energy - (frag1_energy + frag2_energy)
      return torch.unsqueeze(binding_energy, 1)


class AtomicConvModel(OpenChemModel):

  def __init__(self, params):
    super(AtomicConvModel, self).__init__(params)

    frag1_num_atoms, frag2_num_atoms, complex_num_atoms, 
    max_num_neighbors, batch_size, atom_types, radial, 
    layer_sizes, learning_rate = params

    self.complex_num_atoms = complex_num_atoms
    self.frag1_num_atoms = frag1_num_atoms
    self.frag2_num_atoms = frag2_num_atoms
    self.max_num_neighbors = max_num_neighbors
    self.batch_size = batch_size
    self.atom_types = atom_types

    rp = [x for x in itertools.product(*radial)]

    conv1 = AtomicConvolution(
        atom_types=self.atom_types, radial_params=rp,
        boxsize=None)
    self.conv1 = conv1
    conv2 = AtomicConvolution(
        atom_types=self.atom_types, radial_params=rp,
        boxsize=None)
    self.conv2 = conv2
    conv3 = AtomicConvolution(
        atom_types=self.atom_types, radial_params=rp,
        boxsize=None)
    self.conv3 = conv3
    score = AtomicConvScore(self.atom_types, layer_sizes)
    self.score = score

    def forward():
        frag1_X = torch.Tensor(self.frag1_num_atoms, 3)
        frag1_nbrs = torch.Tensor(self.frag1_num_atoms, self.max_num_neighbors)
        frag1_nbrs_z = torch.Tensor(self.frag1_num_atoms, self.max_num_neighbors)
        frag1_z = torch.Tensor(self.frag1_num_atoms,)

        frag2_X = torch.Tensor(self.frag2_num_atoms, 3)
        frag2_nbrs = torch.Tensor(self.frag2_num_atoms, self.max_num_neighbors)
        frag2_nbrs_z = torch.Tensor(self.frag2_num_atoms, self.max_num_neighbors)
        frag2_z = torch.Tensor(self.frag2_num_atoms,)

        complex_X = torch.Tensor(self.complex_num_atoms, 3)
        complex_nbrs = torch.Tensor(self.complex_num_atoms, self.max_num_neighbors)
        complex_nbrs_z = torch.Tensor(self.complex_num_atoms, self.max_num_neighbors)
        complex_z = torch.Tensor(self.complex_num_atoms,)

        frag1_conv = self.conv1(([frag1_X, frag1_nbrs, frag1_nbrs_z]))
        frag2_conv = self.conv2([frag2_X, frag2_nbrs, frag2_nbrs_z])
        complex_conv = self.conv3([complex_X, complex_nbrs, complex_nbrs_z])
        out = self.score([frag1_conv, frag2_conv, complex_conv, frag1_z, frag2_z, complex_z])
        
        return out

    def cast_input(self, sample):
      batch_X = torch.tensor(sample['X_matrix'],
          requires_grad=True).float()
      batch_Nbrs = torch.tensor(sample['Nbrs_matrix'],
          requires_grad=True).float()
      batch_Nbrs_Z = torch.tensor(sample['Nbrs_Z_matrix'],
          requires_grad=True).float()
      batch_labels = torch.tensor(sample['labels'])
      if self.task == 'classification':
          batch_labels = batch_labels.long()
      else:
          batch_labels = batch_labels.float()
      if self.use_cuda:
          print('use_cuda')
          batch_X = batch_X.cuda()
          batch_Nbrs = batch_Nbrs.cuda()
          batch_Nbrs_Z = batch_Nbrs_Z.cuda()
          batch_labels = batch_labels.cuda()
      batch_inp = (batch_X, batch_Nbrs, batch_Nbrs_Z)
      return batch_inp, batch_labels

    def default_generator(self,
                        dataset,
                        epochs=1,
                        mode='fit',
                        deterministic=True,
                        pad_batches=True):
        batch_size = self.batch_size

    def replace_atom_types(z):

      def place_holder(i):
        if i in self.atom_types:
          return i
        return -1

      return np.array([place_holder(x) for x in z])

    for epoch in range(epochs):
      for ind, (F_b, y_b, w_b, ids_b) in enumerate(
          dataset.iterbatches(
              batch_size, deterministic=True, pad_batches=pad_batches)):
        N = self.complex_num_atoms
        N_1 = self.frag1_num_atoms
        N_2 = self.frag2_num_atoms
        M = self.max_num_neighbors

        batch_size = F_b.shape[0]
        num_features = F_b[0][0].shape[1]
        frag1_X_b = np.zeros((batch_size, N_1, num_features))
        for i in range(batch_size):
          frag1_X_b[i] = F_b[i][0]

        frag2_X_b = np.zeros((batch_size, N_2, num_features))
        for i in range(batch_size):
          frag2_X_b[i] = F_b[i][3]

        complex_X_b = np.zeros((batch_size, N, num_features))
        for i in range(batch_size):
          complex_X_b[i] = F_b[i][6]

        frag1_Nbrs = np.zeros((batch_size, N_1, M))
        frag1_Z_b = np.zeros((batch_size, N_1))
        for i in range(batch_size):
          z = replace_atom_types(F_b[i][2])
          frag1_Z_b[i] = z
        frag1_Nbrs_Z = np.zeros((batch_size, N_1, M))
        for atom in range(N_1):
          for i in range(batch_size):
            atom_nbrs = F_b[i][1].get(atom, "")
            frag1_Nbrs[i, atom, :len(atom_nbrs)] = np.array(atom_nbrs)
            for j, atom_j in enumerate(atom_nbrs):
              frag1_Nbrs_Z[i, atom, j] = frag1_Z_b[i, atom_j]

        frag2_Nbrs = np.zeros((batch_size, N_2, M))
        frag2_Z_b = np.zeros((batch_size, N_2))
        for i in range(batch_size):
          z = replace_atom_types(F_b[i][5])
          frag2_Z_b[i] = z
        frag2_Nbrs_Z = np.zeros((batch_size, N_2, M))
        for atom in range(N_2):
          for i in range(batch_size):
            atom_nbrs = F_b[i][4].get(atom, "")
            frag2_Nbrs[i, atom, :len(atom_nbrs)] = np.array(atom_nbrs)
            for j, atom_j in enumerate(atom_nbrs):
              frag2_Nbrs_Z[i, atom, j] = frag2_Z_b[i, atom_j]

        complex_Nbrs = np.zeros((batch_size, N, M))
        complex_Z_b = np.zeros((batch_size, N))
        for i in range(batch_size):
          z = replace_atom_types(F_b[i][8])
          complex_Z_b[i] = z
        complex_Nbrs_Z = np.zeros((batch_size, N, M))
        for atom in range(N):
          for i in range(batch_size):
            atom_nbrs = F_b[i][7].get(atom, "")
            complex_Nbrs[i, atom, :len(atom_nbrs)] = np.array(atom_nbrs)
            for j, atom_j in enumerate(atom_nbrs):
              complex_Nbrs_Z[i, atom, j] = complex_Z_b[i, atom_j]

        inputs = [
            frag1_X_b, frag1_Nbrs, frag1_Nbrs_Z, frag1_Z_b, frag2_X_b,
            frag2_Nbrs, frag2_Nbrs_Z, frag2_Z_b, complex_X_b, complex_Nbrs,
            complex_Nbrs_Z, complex_Z_b
        ]
        y_b = np.reshape(y_b, newshape=(batch_size, 1))
        yield (inputs, [y_b], [w_b])
