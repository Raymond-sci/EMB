from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.layers import mask_logits
from util.runner_utils import calculate_batch_iou

class Conv2dRanker(nn.Module):

    def __init__(self, dim=128, kernel_size=3, num_layers=4, **kwargs):
        super().__init__()
        self.kernel = kernel_size
        self.encoders = nn.ModuleList([
            nn.Conv2d(dim, dim, kernel_size, padding=kernel_size//2) 
        for _ in range(num_layers)])
        self.predictor = nn.Conv2d(dim, 1, 1)

    @staticmethod
    def get_padded_mask_and_weight(mask, conv):
        masked_weight = torch.round(F.conv2d(mask.clone().float(), 
            mask.new_ones(1, 1, *conv.kernel_size),
            stride=conv.stride, padding=conv.padding, dilation=conv.dilation))
        masked_weight[masked_weight > 0] = 1 / masked_weight[masked_weight > 0] #conv.kernel_size[0] * conv.kernel_size[1]
        padded_mask = masked_weight > 0
        return padded_mask, masked_weight

    def forward(self, x, mask):
        # convert to 2d if input is flat
        if x.dim() < 4:
            B, N2, D = x.shape
            assert int(math.sqrt(N2)) == math.sqrt(N2)
            N = int(math.sqrt(N2))
            x2d, mask2d = x.view(B, N, N, D), mask.view(B, N, N)
        else:
            x2d, mask2d = x, mask
        # x: (<bsz>, <num>, <num>, <dim>) -> (<bsz>, <dim>, <num>, <num>)
        x2d, mask2d = x2d.permute(0, 3, 1, 2), mask2d.unsqueeze(1)
        for encoder in self.encoders:
            # mask invalid features to 0
            x2d = F.relu(encoder(x2d * mask2d))
            _, weights = self.get_padded_mask_and_weight(mask2d, encoder)
            x2d = x2d * weights
        # preds: (<bsz>, <num>, <num>)
        preds = self.predictor(x2d).view_as(mask)
        preds = mask_logits(preds, mask)
        return preds.sigmoid()

    @staticmethod
    def topk_confident(bounds, scores, mask, moments=None, threshold=0.5, k=1):
        if moments is not None:
            # compute the overlaps between proposals and ground-truth
            overlaps = calculate_batch_iou(bounds, moments.unsqueeze(1))
            # set the scores of proposals with 
            # insufficient overlaps with the ground-truth to -inf
            is_cand = (overlaps >= threshold) * mask
        else:
            is_cand = mask
        masked_scores = mask_logits(scores, is_cand)
        # get topk proposals
        cands_idx = masked_scores.topk(k, dim=1)[1]
        cands = bounds.gather(1, cands_idx.unsqueeze(-1).repeat(1, 1, 2))
        if moments is not None:
            # in training
            # use the ground-truth moment if there is no proposal whose overlaps 
            # with the ground-truth is greater than the threshold
            # for example, when the threshold equal to 1
            has_cand = (is_cand.sum(-1) > 0).view(-1, 1, 1)
            cands = cands * has_cand + moments.unsqueeze(1) * (~has_cand)
        return cands

    @staticmethod
    def compute_loss(moments, bounds, scores, mask, min_iou=0.3, max_iou=0.7):
        assert scores.shape == mask.shape and min_iou <= max_iou
        ious = calculate_batch_iou(bounds, moments.unsqueeze(1))
        if min_iou == max_iou:
            targets = (ious >= min_iou).float()
        else:
            targets = (ious - min_iou) / (max_iou - min_iou)
            targets = targets.clamp(min=0, max=1)
        return F.binary_cross_entropy(
            scores.masked_select(mask.bool()),
            targets.masked_select(mask.bool()))