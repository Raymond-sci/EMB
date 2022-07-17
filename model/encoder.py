from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.layers import mask_pooling, Conv1D, downscale1d, mask_logits


class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, dim, num_heads=8, drop_rate=0.1):
        super().__init__()
        assert dim % num_heads == 0, ('The hidden size ({}) is not a multiple '
                'of the number of attention heads ({})'.format(dim, num_heads))
        self.num_heads = num_heads
        self.head_size = int(dim / num_heads)

        self.query = Conv1D(in_dim=dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.key = Conv1D(in_dim=dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.value = Conv1D(in_dim=dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)

        self.dropout = nn.Dropout(drop_rate)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_size)  # (N, L, nh, dh)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # (N, nh, L, dh)

    def forward(self, q, k, v, mask):
        # projections
        query = self.transpose_for_scores(self.query(q))
        key = self.transpose_for_scores(self.key(k))
        value = self.transpose_for_scores(self.value(v))
        # attention scores
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.head_size)
        # mask attention scores
        attention_scores = mask_logits(attention_scores, mask.unsqueeze(1))
        # normalise the attention scores to probabilities then dropout randomly
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        # weighted sum of value
        context = torch.matmul(attention_probs, value).permute(0, 2, 1, 3).contiguous()
        # merge features from different heads
        context = context.view(*context.shape[:-2], -1)

        return context


class AttentionLayer(nn.Module):

    def __init__(self, dim, num_heads=8, drop_rate=0.1):
        super().__init__()
        self.attention = MultiHeadAttentionBlock(dim, num_heads, drop_rate)
        self.dense = nn.Sequential(Conv1D(in_dim=dim, out_dim=dim), 
                    nn.ReLU(inplace=True), Conv1D(in_dim=dim, out_dim=dim))
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, query, key=None, value=None, mask=None):
        assert (key is None and value is None) or (key is not None and value is not None)
        # when k, v is None, then is doing self-attention 
        # and the mask is expected to be a 2D matrix (<bsz>, <#query>)
        if key is None and value is None and mask is not None: mask = mask.unsqueeze(1)
        # query is taken as both key and value for self-attention
        if key is None: key = query
        if value is None: value = query
        # if mask is None then all elements in key is valid
        if mask is None: mask = key.new_ones(*key.shape[:-1]).unsqueeze(1)

        # apply attention
        context = self.attention(query, key, value, mask)
        # drop-add-norm
        residual = self.norm1(query + self.dropout(context))
        # apply FFN
        hidden = self.dense(residual)
        # drop-add-norm
        hidden = self.norm2(residual + self.dropout(hidden))
        return hidden


class SelfAttentionBlock(nn.Module):

    def __init__(self, dim=128, num_layers=1, num_heads=8, drop_rate=0.1, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([AttentionLayer(dim, num_heads, drop_rate) for _ in range(num_layers)])

    def forward(self, target=None, tmask=None, **kwargs):
        assert target is not None
        for layer in self.layers:
            target = layer(query=target, mask=tmask)
        return target


class CrossAttentionBlock(nn.Module):

    def __init__(self, dim=128, num_layers=1, num_heads=8, drop_rate=0.1, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([AttentionLayer(dim, num_heads, drop_rate) for _ in range(num_layers)])

    def forward(self, source=None, target=None, smask=None, tmask=None, **kwargs):
        assert None not in [source, target]
        if None not in [smask, tmask]:
            mask = tmask.unsqueeze(2) * smask.unsqueeze(1)
        else:
            mask = source.new_ones(*target.shape[:2], source.shape[1])
        for layer in self.layers:
            target = layer(query=target, key=source, value=source, mask=mask)
        return target


class DepthwiseSeparableConv2dBlock(nn.Module):
    
    def __init__(self, dim, kernel_size, drop_rate, padding):
        super().__init__()
        self.conv_layers = nn.Sequential(
                nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=kernel_size, groups=dim,
                          padding=padding, bias=False),
                nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0, bias=True))
        self.relu = nn.ReLU(inplace=True)
        self.layer_norm = nn.LayerNorm(dim, eps=1e-6)
        self.dropout = nn.Dropout(p=drop_rate)

    def forward(self, x):
        # (batch_size, seq_len, dim)
        output = self.conv_layers(x.transpose(1, 2))
        output = self.dropout(output.transpose(1, 2))
        output = self.layer_norm(output.squeeze(-1))
        output = self.relu(output)
        return output


class GuidedAttentionLayer(nn.Module):

    def __init__(self, dim, num_guides=2, kernel_size=7, num_heads=8, drop_rate=0.1):
        super().__init__()
        self.attention = MultiHeadAttentionBlock(dim, num_heads, drop_rate)
        self.guide = DepthwiseSeparableConv2dBlock(dim, 
            kernel_size=(kernel_size, 1 + num_guides), 
            drop_rate=drop_rate, padding=(kernel_size // 2, 0))
        self.dense = nn.Sequential(Conv1D(in_dim=dim, out_dim=dim), 
                    nn.ReLU(inplace=True), Conv1D(in_dim=dim, out_dim=dim))
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, guide, query, key=None, value=None, mask=None):
        assert (key is None and value is None) or (key is not None and value is not None)
        # concate guide with query features then fuse the query and guided by convolution
        guided = torch.cat([guide, query.unsqueeze(-1)], dim=-1)
        guided = self.guide(guided)
        # guided = self.dropout(query + guided)

        # when k, v is None, then is doing self-attention
        if key is None and value is None and mask is not None: mask = mask.unsqueeze(1)
        # query is taken as both key and value for self-attention
        if key is None: key = guided
        if value is None: value = guided
        # if mask is None then all elements in key is valid
        if mask is None: mask = key.new_ones(*key.shape[:-1]).unsqueeze(1)

        # apply attention
        context = self.attention(guided, key, value, mask)
        # drop-add-norm
        residual = self.norm1(query + self.dropout(context))
        # apply FFN
        hidden = self.dense(residual)
        # drop-add-norm
        hidden = self.norm2(residual + self.dropout(hidden))
        return hidden


class ContentGuidedAttentionLayerAdaptor(GuidedAttentionLayer):

    def __init__(self, dim, num_heads=8, drop_rate=0.1):
        super().__init__(dim=dim, num_guides=2, num_heads=num_heads, drop_rate=drop_rate)

    def forward(self, query, key=None, value=None, mask=None):
        # query: (<bsz>, <#frames>, <dim>) -> guide: (<bsz>, <#frames>, <dim>, 2)
        # max-pooling all the frames before and after every frame as the guide of
        # the frame itself
        # indices: (1, <#frames>), guides: (<bsz>, 1, <#frames>, <dim>)
        indices = torch.arange(query.shape[1], device=query.device).view(1, -1)
        guides = query.unsqueeze(1)
        qmask = query.new_ones(*query.shape[:2])
        if mask is not None:
            if mask.dim() == 2: qmask = mask
            elif mask.dim() == 3: qmask = (mask > 0).any(-1).float()
            else: raise ValueError('Mask is expected to be either 2 or 3 dims')
        qmask2d = qmask.unsqueeze(2) * qmask.unsqueeze(1)
        # lmask/rmask: (1, <#frames>, <#frames>)
        lmask = (indices.unsqueeze(2) > indices.unsqueeze(1)) * qmask2d
        rmask = (indices.unsqueeze(2) < indices.unsqueeze(1)) * qmask2d
        # lguide/rguide: (<bsz>, <#frames>, <dim>)
        lguide = mask_pooling(guides, lmask.unsqueeze(-1), mode='max', dim=2)
        rguide = mask_pooling(guides, rmask.unsqueeze(-1), mode='max', dim=2)
        # if any guide contain non items, then set its feature to all zeros
        lguide *= (lmask.sum(-1) > 0).unsqueeze(-1)
        rguide *= (rmask.sum(-1) > 0).unsqueeze(-1)
        guide = torch.stack([lguide, rguide], dim=-1)
        # mask all invalid features in query to 0
        query *= qmask.unsqueeze(-1)
        return super().forward(guide, query, key=key, value=value, mask=mask)


class BoundGuidedAttentionLayerAdaptor(GuidedAttentionLayer):

    def __init__(self, dim, num_heads=8, drop_rate=0.1, downscale=8):
        super().__init__(dim=dim, kernel_size=1, num_guides=2, num_heads=num_heads, drop_rate=drop_rate)
        self.scale = downscale

    def forward(self, guide, lindices, rindices, query, key=None, value=None, mask=None):
        # downscale frames features
        guide = downscale1d(guide.transpose(1, 2), scale=self.scale, mode='max')
        guide = guide.transpose(1, 2)
        lindices = lindices // self.scale
        rindices = rindices // self.scale
        # get guide features according to indices
        lguide = guide.gather(1, lindices.unsqueeze(-1).repeat(1, 1, guide.shape[-1]))
        rguide = guide.gather(1, rindices.unsqueeze(-1).repeat(1, 1, guide.shape[-1]))
        # guess query mask
        qmask = query.new_ones(*query.shape[:2])
        if mask is not None:
            if mask.dim() == 2: qmask = mask
            elif mask.dim() == 3: qmask = (mask > 0).any(-1).float()
            else: raise ValueError('Mask is expected to be either 2 or 3 dims')
        guide = torch.stack([lguide, rguide], dim=-1)
        guide *= qmask.unsqueeze(-1).unsqueeze(-1)
        query *= qmask.unsqueeze(-1)
        return super().forward(guide, query, key=key, value=value, mask=mask)


class ContentGuidedSelfAttentionBlock(nn.Module):

    def __init__(self, dim=128, num_layers=1, num_heads=8, drop_rate=0.1, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([ContentGuidedAttentionLayerAdaptor(dim, 
                    num_heads, drop_rate) for _ in range(num_layers)])

    def forward(self, target=None, tmask=None, **kwargs):
        assert target is not None
        for layer in self.layers:
            target = layer(query=target, mask=tmask)
        return target


class ContentGuidedCrossAttentionBlock(nn.Module):

    def __init__(self, dim=128, num_layers=1, num_heads=8, drop_rate=0.1, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([ContentGuidedAttentionLayerAdaptor(dim, 
                    num_heads, drop_rate) for _ in range(num_layers)])

    def forward(self, source=None, target=None, smask=None, tmask=None, **kwargs):
        assert None not in [source, target]
        if None not in [smask, tmask]:
            mask = tmask.unsqueeze(2) * smask.unsqueeze(1)
        else:
            mask = source.new_ones(*target.shape[:2], source.shape[1])
        for layer in self.layers:
            target = layer(query=target, key=source, value=source, mask=mask)
        return target


class BoundGuidedSelfAttentionBlock(nn.Module):

    def __init__(self, dim=128, num_layers=1, num_heads=8, drop_rate=0.1, downscale=8, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([BoundGuidedAttentionLayerAdaptor(dim, 
                    num_heads, drop_rate, downscale) for _ in range(num_layers)])

    def forward(self, target=None, tmask=None, guide=None, bounds=None, **kwargs):
        assert None not in [target, guide, bounds]
        for layer in self.layers:
            target = layer(guide=guide, lindices=bounds[...,0], 
                        rindices=bounds[...,1], query=target, mask=tmask,)
        return target


class BoundGuidedCrossAttentionBlock(nn.Module):

    def __init__(self, dim=128, num_layers=1, num_heads=8, drop_rate=0.1, downscale=8, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([BoundGuidedAttentionLayerAdaptor(dim, 
                    num_heads, drop_rate, downscale) for _ in range(num_layers)])

    def forward(self, source=None, target=None, smask=None, tmask=None, guide=None, bounds=None, **kwargs):
        assert None not in [source, target, guide, bounds]
        if None not in [smask, tmask]:
            mask = tmask.unsqueeze(2) * smask.unsqueeze(1)
        else:
            mask = source.new_ones(*target.shape[:2], source.shape[1])
        for layer in self.layers:
            target = layer(guide=guide, lindices=bounds[...,0], rindices=bounds[...,1],
                            query=target, key=source, value=source, mask=mask)
        return target


class GuidedAttentionEncoder(nn.Module):

    def __init__(self, dim=128, num_layers=1, num_heads=8, drop_rate=0.1, downscale=8, 
                proposal=None, **kwargs):
        super().__init__()
        assert proposal is not None
        self.proposal = proposal
        self.f2f = ContentGuidedSelfAttentionBlock(dim=dim, num_layers=num_layers,
            num_heads=num_heads, drop_rate=drop_rate)
        self.p2p = BoundGuidedSelfAttentionBlock(dim=dim, num_layers=num_layers,
            num_heads=num_heads, drop_rate=drop_rate, downscale=downscale)
        self.w2w = SelfAttentionBlock(dim=dim, num_layers=num_layers,
            num_heads=num_heads, drop_rate=drop_rate)
        self.f2w = ContentGuidedCrossAttentionBlock(dim=dim, num_layers=num_layers,
            num_heads=num_heads, drop_rate=drop_rate)
        self.p2w = BoundGuidedCrossAttentionBlock(dim=dim, num_layers=num_layers,
            num_heads=num_heads, drop_rate=drop_rate)
        self.w2f = CrossAttentionBlock(dim=dim, num_layers=num_layers,
            num_heads=num_heads, drop_rate=drop_rate)
        self.w2p = copy.deepcopy(self.w2f)
        self.v2v = copy.deepcopy(self.f2f)
        self.pq2q = copy.deepcopy(self.w2w)
        self.vq2q = copy.deepcopy(self.w2w)
        self.s2s = copy.deepcopy(self.p2p)

    def forward(self, vfeats=None, qfeats=None, vmask=None, qmask=None, moments=None, **kwargs):
        assert None not in [vfeats, qfeats]
        pfeats, proposals, pmask = self.proposal(feats=vfeats, mask=vmask, moments=moments)
        pfeats, vfeats, qfeats = self.p2p(pfeats, pmask, guide=vfeats, bounds=proposals), self.f2f(vfeats, vmask), self.w2w(qfeats, qmask)
        pfeats, pqfeats = self.p2w(qfeats, pfeats, qmask, pmask, guide=vfeats, bounds=proposals), self.w2p(pfeats, qfeats, pmask, qmask)
        vfeats, vqfeats = self.f2w(qfeats, vfeats, qmask, vmask), self.w2f(vfeats, qfeats, vmask, qmask)
        pfeats, pqfeats = self.s2s(pfeats, pmask, guide=vfeats, bounds=proposals), self.pq2q(pqfeats, qmask)
        vfeats, vqfeats = self.v2v(vfeats, vmask), self.vq2q(vqfeats, qmask)
        return (vfeats, vqfeats), (pfeats, pqfeats), proposals, pmask
