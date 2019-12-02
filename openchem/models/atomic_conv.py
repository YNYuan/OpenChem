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


class AtomicConvModel(OpenChemModel):

  def __init__(self, params):
    super(AtomicConvModel, self).__init__(params)

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
    ligand_X = input[0] 
    ligand_Z = input[1]
    ligand_R = input[2]
    ligand_Nbrs_Z = input[3]
    
    ligand_conv = self.conv_l.forward([ligand_X, ligand_R, ligand_Nbrs_Z])

    ligand_zeros = torch.zeros_like(ligand_Z)
    ligand_atomtype_energy = []
    for ind, atomtype in enumerate(self.atom_types):
      if self.use_cuda:
        ligand_outputs = torch.FloatTensor(ligand_conv.shape[0], ligand_conv.shape[1], 1).cuda()
      else:
        ligand_outputs = torch.FloatTensor(ligand_conv.shape[0], ligand_conv.shape[1], 1)
      for i, sample in enumerate(ligand_conv):
        ligand_outputs[i] = self.MLP_list[ind](sample)
      cond = torch.eq(ligand_Z, atomtype)
      ligand_atomtype_energy.append(
        torch.where(cond, ligand_outputs, ligand_zeros))
    ligand_outputs = torch.stack(ligand_atomtype_energy, dim=0).sum(dim=0)
    ligand_energy = torch.sum(ligand_outputs, 1)
    if self.use_cuda:
      ligand_energy = ligand_energy.cuda()
    return ligand_energy#torch.unsqueeze(complex_energy, 1)

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
        batch_X = batch_X_c.cuda()
        batch_Z = batch_Z_c.cuda()
        #batch_Nbrs = batch_Nbrs.cuda()
        batch_R = batch_R_c.cuda()
        batch_Nbrs_Z = batch_Nbrs_Z_c.cuda()
        batch_labels = batch_labels.cuda()
    batch_inp = (batch_X, batch_Z, batch_R, batch_Nbrs_Z)
    return batch_inp, batch_labels
