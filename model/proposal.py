from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.layers import mask_logits, downscale1d


class TAN2dProposal(nn.Module):

    def __init__(self, downscale=8, windows=[16], **kwargs):
        super().__init__()
        # downscale before generating proposals
        self.scale = downscale
        # pooling layers
        self.windows = windows
        layers = []
        for i, window in enumerate(windows):
            layers.extend([nn.MaxPool1d(1, 1) if i == 0 else nn.MaxPool1d(3, 2)])
            layers.extend([nn.MaxPool1d(2, 1) for _ in range(window - 1)])
        self.layers = nn.ModuleList(layers)

    def forward(self, feats=None, mask=None, **kwargs):
        assert None not in [feats, mask]
        # set all invalid features to a very small values
        # to avoid their impact in maxpooling
        feats = mask_logits(feats, mask.unsqueeze(-1))
        # apply downscale first
        feats = downscale1d(feats.transpose(1, 2), scale=self.scale, mode='max')
        scaled_mask = downscale1d(mask.unsqueeze(1), scale=self.scale, mode='max').squeeze(1)
        B, D, N = feats.shape
        mask2d = mask.new_zeros(B, N, N)
        feat2d = feats.new_zeros(B, D, N, N)
        offset, stride = -1, 1
        for i, window in enumerate(self.windows):
            for j in range(window):
                layer = self.layers[i * len(self.windows) + j]
                if feats.shape[-1] < layer.kernel_size: break
                offset += stride
                start, end = range(0, N - offset, stride), range(offset, N, stride)
                # assume valid features are continual
                mask2d[:, start, end] = scaled_mask[:, end]
                feats = layer(feats)
                feat2d[:,:,start,end] = feats
            stride *= 2
        # mask invalid proposal features to 0
        feat2d *= mask2d.unsqueeze(1)
        # (B, D, N, N) -> (B, N, N, D)
        feat2d = feat2d.permute(0, 2, 3, 1)
        # generate boundary
        bounds = torch.arange(0, N, device=feats.device)
        bounds = bounds.view(1, -1).repeat(B, 1)
        # (B, N, N, 2)
        bounds = torch.stack([bounds.unsqueeze(-1).repeat(1, 1, N) * self.scale,
            (bounds.unsqueeze(1).repeat(1, N, 1) + 1) * self.scale - 1], dim=-1)
        # set the largest boundary to the number of items in each sample
        bounds = torch.min(bounds, mask.sum(-1).view(-1, 1, 1, 1).long() - 1)
        # mask invalid proposal
        bounds *= mask2d.unsqueeze(-1).long()
        # make sure for all the valid proposals
        # its endpoint should be greater or equal to its start points
        assert ((bounds[...,1] >= bounds[...,0]) + ~mask2d.bool()).all()
        # flatten all proposals as output
        feat2d = feat2d.view(B, N * N, D)
        bounds = bounds.view(B, N * N, 2)
        mask2d = mask2d.view(B, N * N)
        return feat2d, bounds, mask2d
