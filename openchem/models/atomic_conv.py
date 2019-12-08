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
    ligand_X = input[0] 
    ligand_Z = input[1]
    ligand_R = input[2]
    ligand_Nbrs_Z = input[3]
    
    ligand_conv = self.conv.forward([ligand_X, ligand_R, ligand_Nbrs_Z])

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
    return ligand_energy

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

    conv_l = AtomicConvolution(
        atom_types=self.atom_types, radial_params=rp,
        boxsize=None, use_cuda=self.use_cuda)
    self.conv_l = conv_l
    conv_r = AtomicConvolution(
        atom_types=self.atom_types, radial_params=rp,
        boxsize=None, use_cuda=self.use_cuda)
    self.conv_r = conv_r
    conv_c = AtomicConvolution(
        atom_types=self.atom_types, radial_params=rp,
        boxsize=None, use_cuda=self.use_cuda)
    self.conv_c = conv_c

  def forward(self, input, eval=False):
    if eval:
      self.eval()
    else:
      self.train()
    ligand_X = input[0] 
    ligand_Z = input[1]
    ligand_R = input[2]
    ligand_Nbrs_Z = input[3]
    rec_X = input[4] 
    rec_Z = input[5]
    rec_R = input[6]
    rec_Nbrs_Z = input[7]
    complex_X = input[8] 
    complex_Z = input[9]
    complex_R = input[10]
    complex_Nbrs_Z = input[11]
    
    ligand_conv = self.conv_l.forward([ligand_X, ligand_R, ligand_Nbrs_Z])
    rec_conv = self.conv_r.forward([rec_X, rec_R, rec_Nbrs_Z])
    complex_conv = self.conv_c.forward([complex_X, complex_R, complex_Nbrs_Z])

    ligand_zeros = torch.zeros_like(ligand_Z)
    ligand_atomtype_energy = []
    rec_zeros = torch.zeros_like(rec_Z)
    rec_atomtype_energy = []
    complex_zeros = torch.zeros_like(complex_Z)
    complex_atomtype_energy = []
    for ind, atomtype in enumerate(self.atom_types):
      if self.use_cuda:
        ligand_outputs = torch.FloatTensor(ligand_conv.shape[0], ligand_conv.shape[1], 1).cuda()
        rec_outputs = torch.FloatTensor(rec_conv.shape[0], rec_conv.shape[1], 1).cuda()
        complex_outputs = torch.FloatTensor(complex_conv.shape[0], complex_conv.shape[1], 1).cuda()
      else:
        ligand_outputs = torch.FloatTensor(ligand_conv.shape[0], ligand_conv.shape[1], 1)
        rec_outputs = torch.FloatTensor(rec_conv.shape[0], rec_conv.shape[1], 1)
        complex_outputs = torch.FloatTensor(complex_conv.shape[0], complex_conv.shape[1], 1)
      for i, sample in enumerate(ligand_conv):
        ligand_outputs[i] = self.MLP_list[ind](sample)
      for i, sample in enumerate(rec_conv):
        rec_outputs[i] = self.MLP_list[ind](sample)
      for i, sample in enumerate(complex_conv):
        complex_outputs[i] = self.MLP_list[ind](sample)
      cond = torch.eq(ligand_Z, atomtype)
      ligand_atomtype_energy.append(
        torch.where(cond, ligand_outputs, ligand_zeros))
      cond = torch.eq(rec_Z, atomtype)
      rec_atomtype_energy.append(
        torch.where(cond, rec_outputs, rec_zeros))
      cond = torch.eq(complex_Z, atomtype)
      complex_atomtype_energy.append(
        torch.where(cond, complex_outputs, complex_zeros))
    ligand_outputs = torch.stack(ligand_atomtype_energy, dim=0).sum(dim=0)
    rec_outputs = torch.stack(rec_atomtype_energy, dim=0).sum(dim=0)
    complex_outputs = torch.stack(complex_atomtype_energy, dim=0).sum(dim=0)
    ligand_energy = torch.sum(ligand_outputs, 1)
    rec_energy = torch.sum(rec_outputs, 1)
    complex_energy = torch.sum(complex_outputs, 1)
    binding_energy = complex_energy - ligand_energy - rec_energy
    if self.use_cuda:
      binding_energy = binding_energy.cuda()
    return binding_energy

  def cast_inputs(self, sample):
    batch_X_l = sample['X_matrix_l'].clone().detach().requires_grad_(True)
    batch_Z_l = sample['Z_matrix_l'].clone().detach().requires_grad_(True)
    # batch_Nbrs = torch.tensor(sample['Nbrs_matrix'],
    #     requires_grad=True).float()
    batch_R_l = sample['R_matrix_l'].clone().detach().requires_grad_(True)
    batch_Nbrs_Z_l = sample['Nbrs_Z_matrix_l'].clone().detach().requires_grad_(True)
    batch_X_r = sample['X_matrix_r'].clone().detach().requires_grad_(True)
    batch_Z_r = sample['Z_matrix_r'].clone().detach().requires_grad_(True)
    batch_R_r = sample['R_matrix_r'].clone().detach().requires_grad_(True)
    batch_Nbrs_Z_r = sample['Nbrs_Z_matrix_r'].clone().detach().requires_grad_(True)
    batch_X_c = sample['X_matrix_c'].clone().detach().requires_grad_(True)
    batch_Z_c = sample['Z_matrix_c'].clone().detach().requires_grad_(True)
    batch_R_c = sample['R_matrix_c'].clone().detach().requires_grad_(True)
    batch_Nbrs_Z_c = sample['Nbrs_Z_matrix_c'].clone().detach().requires_grad_(True)
    batch_labels = sample['target'].clone().detach().requires_grad_(False)
    if self.task == 'classification':
        batch_labels = batch_labels.long()
    else:
        batch_labels = batch_labels.float()
    if self.use_cuda:
        batch_X_l = batch_X_l.cuda()
        batch_Z_l = batch_Z_l.cuda()
        #batch_Nbrs = batch_Nbrs.cuda()
        batch_R_l = batch_R_l.cuda()
        batch_Nbrs_Z_l = batch_Nbrs_Z_l.cuda()
        batch_X_r = batch_X_r.cuda()
        batch_Z_r = batch_Z_r.cuda()
        batch_R_r = batch_R_r.cuda()
        batch_Nbrs_Z_r = batch_Nbrs_Z_r.cuda()
        batch_X_c = batch_X_c.cuda()
        batch_Z_c = batch_Z_c.cuda()
        batch_R_c = batch_R_c.cuda()
        batch_Nbrs_Z_c = batch_Nbrs_Z_c.cuda()
        batch_labels = batch_labels.cuda()
    batch_inp = (batch_X_l, batch_Z_l, batch_R_l, batch_Nbrs_Z_l, batch_X_r, batch_Z_r, 
    batch_R_r, batch_Nbrs_Z_r, batch_X_c, batch_Z_c, batch_R_c, batch_Nbrs_Z_c)
    return batch_inp, batch_labels