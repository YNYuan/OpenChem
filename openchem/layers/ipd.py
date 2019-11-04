# modified from https://github.com/tkipf/gae/blob/master/gae/layers.py

import torch
import torch.nn as nn

class InnerProductDecoder(nn.Module):
    """Decoder model layer for link prediction."""
    def __init__(self, input_dim, act=nn.functional.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.act = act

    def forward(self, inputs):
        x = inputs.transpose(1,2)
        x = torch.mm(inputs, x)
        x = x.view(-1)
        outputs = self.act(x)
        return outputs