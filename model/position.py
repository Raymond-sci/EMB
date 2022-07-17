from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FixedPosition(nn.Module):
    """
    Add positional information to input tensor.
    :Examples:
        >>> model = PositionEncoding(n_filters=6, max_len=10)
        >>> test_input1 = torch.zeros(3, 10, 6)
        >>> output1 = model(test_input1)
        >>> output1.size()
        >>> test_input2 = torch.zeros(5, 3, 9, 6)
        >>> output2 = model(test_input2)
        >>> output2.size()
    """

    def __init__(self, dim=128, max_pos_len=128, pe_type="cosine"):
        """
        :param n_filters: same with input hidden size
        :param max_len: maximum sequence length
        :param pe_type: cosine or linear or None
        """
        super().__init__()
        self.pe_type = pe_type
        if pe_type != "none":
            position = torch.arange(0, max_pos_len).float().unsqueeze(1)
            if pe_type == "cosine":
                # Compute the positional encodings once in log space.
                pe = torch.zeros(max_pos_len, dim)  # (L, D)
                div_term = torch.exp(torch.arange(0, dim, 2).float() * - (math.log(10000.0) / dim))
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
            elif pe_type == "linear":
                pe = position / max_pos_len
            else:
                raise ValueError
            self.register_buffer("pe", pe)  # buffer is a tensor, not a variable, (L, D)

    def forward(self, x):
        """
        :Input: (*, L, D)
        :Output: (*, L, D) the same size as input
        """
        if self.pe_type != "none":
            pe = self.pe.data[:x.size(-2), :]  # (#x.size(-2), n_filters)
            extra_dim = len(x.size()) - 2
            for _ in range(extra_dim):
                pe = pe.unsqueeze(0)
            x = x + pe
        return x


class CosinePosition(FixedPosition):

    def __init__(self, dim=128, max_pos_len=128, **kwargs):
        super().__init__(dim=dim, max_pos_len=max_pos_len, pe_type='cosine')


class LinearPosition(FixedPosition):

    def __init__(self, dim=128, max_pos_len=128, **kwargs):
        super().__init__(dim=dim, max_pos_len=max_pos_len, pe_type='linear')