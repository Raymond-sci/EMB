from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from model.layers import Conv1D


class GeneralRNNBlock(nn.Module):

    def __init__(self, word_embedding_size, hidden_size, bidirectional=True,
                 dropout_p=0, n_layers=1, rnn_type="lstm",
                 return_hidden=True, return_outputs=True,
                 allow_zero=False):
        super().__init__()
        """  
        :param word_embedding_size: rnn input size
        :param hidden_size: rnn output size
        :param dropout_p: between rnn layers, only useful when n_layer >= 2
        """
        self.allow_zero = allow_zero
        self.rnn_type = rnn_type
        self.n_dirs = 2 if bidirectional else 1
        # - add return_hidden keyword arg to reduce computation if hidden is not needed.
        self.return_hidden = return_hidden
        self.return_outputs = return_outputs
        self.rnn = getattr(nn, rnn_type.upper())(word_embedding_size, hidden_size, n_layers,
                                                 batch_first=True,
                                                 bidirectional=bidirectional,
                                                 dropout=dropout_p)

    def sort_batch(self, seq, lengths):
        sorted_lengths, perm_idx = lengths.sort(0, descending=True)
        if self.allow_zero:  # deal with zero by change it to one.
            sorted_lengths[sorted_lengths == 0] = 1
        reverse_indices = [0] * len(perm_idx)
        for i in range(len(perm_idx)):
            reverse_indices[perm_idx[i]] = i
        sorted_seq = seq[perm_idx]
        return sorted_seq, list(sorted_lengths), reverse_indices

    def forward(self, inputs, lengths):
        """
        inputs, sorted_inputs -> (B, T, D)
        lengths -> (B, )
        outputs -> (B, T, n_dirs * D)
        hidden -> (n_layers * n_dirs, B, D) -> (B, n_dirs * D)  keep the last layer
        - add total_length in pad_packed_sequence for compatiblity with nn.DataParallel, --remove it
        """
        assert len(inputs) == len(lengths)
        sorted_inputs, sorted_lengths, reverse_indices = self.sort_batch(inputs, lengths)
        packed_inputs = pack_padded_sequence(sorted_inputs, sorted_lengths, batch_first=True)
        outputs, hidden = self.rnn(packed_inputs)
        if self.return_outputs:
            # outputs, lengths = pad_packed_sequence(outputs, batch_first=True, total_length=int(max(lengths)))
            outputs, lengths = pad_packed_sequence(outputs, batch_first=True)
            outputs = outputs[reverse_indices]
        else:
            outputs = None
        if self.return_hidden:  #
            if self.rnn_type.lower() == "lstm":
                hidden = hidden[0]
            hidden = hidden[-self.n_dirs:, :, :]
            hidden = hidden.transpose(0, 1).contiguous()
            hidden = hidden.view(hidden.size(0), -1)
            hidden = hidden[reverse_indices]
        else:
            hidden = None
        return outputs, hidden


class GeneralRNNBlockAdaptor(GeneralRNNBlock):

    def __init__(self, word_embedding_size, hidden_size, bidirectional=True,
                 dropout_p=0, n_layers=1, rnn_type="lstm",
                 return_hidden=False, return_outputs=True,
                 allow_zero=False, **kwargs):
        super().__init__(word_embedding_size, hidden_size, bidirectional=bidirectional,
            dropout_p=dropout_p, n_layers=n_layers, rnn_type=rnn_type,
            return_outputs=return_outputs, return_hidden=return_hidden,
            allow_zero=allow_zero)
        # post-processing, if bidirectional is set to True
        # then the returned tensors will be in different size as other encoders
        # so apply an extra fully-connected layer to it
        self.dense_output = None
        if return_outputs and bidirectional:
            self.dense_output = Conv1D(in_dim=2*hidden_size, out_dim=hidden_size, 
                                kernel_size=1, stride=1, padding=0, bias=True)
        self.dense_hidden = None
        if return_hidden and bidirectional:
            self.dense_hidden = Conv1D(in_dim=2*hidden_size, out_dim=hidden_size,
                                kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, target=None, tmask=None, **kwargs):
        assert target is not None
        if tmask is not None:
            lengths = tmask.sum(-1).view(-1)
        else:
            lengths = target.new_zeros(target.shape[0]) + target.shape[1]
        # get output from parent
        outputs, hidden = super().forward(target, lengths)
        # project the features back to the hidden_size if bidirectional
        if self.dense_output is not None:
            outputs = self.dense_output(outputs)
        if self.dense_hidden is not None:
            hidden = self.dense_hidden(hidden.unsqueeze(1)).squeeze(1)
        if self.return_outputs and not self.return_hidden:
            return outputs
        if self.return_hidden and not self.return_outputs:
            return hidden
        return outputs, hidden


class LSTMBlock(GeneralRNNBlockAdaptor):

    def __init__(self, dim=128, drop_rate=0., num_layers=1, bidirectional=True, **kwargs):
        super().__init__(dim, dim, bidirectional=bidirectional,
            dropout_p=drop_rate, n_layers=num_layers, rnn_type='lstm', 
            allow_zero=False, **kwargs)