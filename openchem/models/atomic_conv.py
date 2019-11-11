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


class AtomicConvModel0(OpenChemModel):

  def __init__(self, params):
    super(AtomicConvModel0, self).__init__(params)

    self.radial = params['radial']
    self.atom_types = params['atom_types']
    self.MLP_list = nn.ModuleList([])
    for ind, atomtype in enumerate(self.atom_types):
      self.mlp = self.params['mlp']
      self.mlp_params = self.params['mlp_params']
      if self.use_cuda:
        self.MLP = self.mlp(self.mlp_params).cuda()
      else:
        self.MLP = self.mlp(self.mlp_params)
      self.MLP_list.append(self.MLP)

    rp = [x for x in itertools.product(*self.radial)]

    conv = AtomicConvolution(
        atom_types=self.atom_types, radial_params=rp,
        boxsize=None, use_cuda=self.use_cuda)
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
      if self.use_cuda:
        complex_outputs = torch.FloatTensor(complex_conv.shape[0], complex_conv.shape[1], 1).cuda()
      else:
        complex_outputs = torch.FloatTensor(complex_conv.shape[0], complex_conv.shape[1], 1)
      for i, sample in enumerate(complex_conv):
        complex_outputs[i] = self.MLP_list[ind](sample)
      cond = torch.eq(complex_Z, atomtype)
      complex_atomtype_energy.append(
        torch.where(cond, complex_outputs, complex_zeros))
    complex_outputs = torch.stack(complex_atomtype_energy, dim=0).sum(dim=0)
    complex_energy = torch.sum(complex_outputs, 1)
    if self.use_cuda:
      complex_energy = complex_energy.cuda()
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
