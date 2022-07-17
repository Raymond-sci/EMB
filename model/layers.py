import copy
import math
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F

from util.runner_utils import calculate_batch_iou


def mask_logits(inputs, mask, mask_value=-1e30):
    mask = mask.type(torch.float32)
    return inputs + (1.0 - mask) * mask_value


def mask_pooling(ntensor, nmask, mode='avg', dim=-1):
    """mark pooling over the given tensor
    """
    if mode in ['avg', 'sum']:
        ntensor = (ntensor * nmask).sum(dim)
        if mode == 'avg':
            ntensor /= nmask.sum(dim).clamp(min=1e-10)
    elif mode == 'max':
        ntensor = mask_logits(ntensor, nmask)
        ntensor = ntensor.max(dim)[0]
    else:
        raise NotImplementedError('Only sum/average/max pooling have been implemented')
    return ntensor


def downscale1d(x, scale=1, dim=None, mode='max'):
    # pad at tail
    padding = (scale - x.shape[-1] % scale) % scale
    x = F.pad(x, (0, padding), value=(0 if mode=='sum' else -1e-30))
    # downscale
    if mode == 'sum':
        kernel = x.new_ones(1, 1, scale)
        return F.conv1d(x, kernel, stride=scale, padding=0)
    elif mode == 'max':
        return F.max_pool1d(x, scale, stride=scale, padding=0)
    raise NotImplementedError('downscale1d implemented only "max" and "sum" mode')


class Conv1D(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=True):
        super(Conv1D, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=in_dim, out_channels=out_dim, kernel_size=kernel_size, padding=padding,
                                stride=stride, bias=bias)

    def forward(self, x):
        # suppose all the input with shape (batch_size, seq_len, dim)
        x = x.transpose(1, 2)  # (batch_size, dim, seq_len)
        x = self.conv1d(x)
        return x.transpose(1, 2)  # (batch_size, seq_len, dim)


class DepthwiseSeparableConvBlock(nn.Module):
    def __init__(self, dim, kernel_size, drop_rate, num_layers=4):
        super(DepthwiseSeparableConvBlock, self).__init__()
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=kernel_size, groups=dim,
                          padding=kernel_size // 2, bias=False),
                nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0, bias=True),
                nn.ReLU(),
            ) for _ in range(num_layers)])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(dim, eps=1e-6) for _ in range(num_layers)])
        self.dropout = nn.Dropout(p=drop_rate)

    def forward(self, x):
        output = x  # (batch_size, seq_len, dim)
        for idx, conv_layer in enumerate(self.conv_layers):
            residual = output
            output = self.layer_norms[idx](output)  # (batch_size, seq_len, dim)
            output = output.transpose(1, 2)  # (batch_size, dim, seq_len)
            output = conv_layer(output)
            output = self.dropout(output)
            output = output.transpose(1, 2) + residual  # (batch_size, seq_len, dim)
        return output


class WordEmbedding(nn.Module):
    def __init__(self, num_words, word_dim, drop_rate, to_norm=False, word_vectors=None):
        super(WordEmbedding, self).__init__()
        self.is_pretrained = False if word_vectors is None else True
        if self.is_pretrained:
            self.pad_vec = nn.Parameter(torch.zeros(size=(1, word_dim), dtype=torch.float32), requires_grad=False)
            unk_vec = torch.empty(size=(1, word_dim), requires_grad=True, dtype=torch.float32)
            nn.init.xavier_uniform_(unk_vec)
            self.unk_vec = nn.Parameter(unk_vec, requires_grad=True)
            self.glove_vec = nn.Parameter(torch.tensor(word_vectors, dtype=torch.float32), requires_grad=False)
        else:
            self.word_emb = nn.Embedding(num_words, word_dim, padding_idx=0)
        self.dropout = nn.Dropout(p=drop_rate)
        self.to_norm = to_norm

    def forward(self, word_ids):
        if self.is_pretrained:
            word_emb = F.embedding(word_ids, torch.cat([self.pad_vec, self.unk_vec, self.glove_vec], dim=0),
                                   padding_idx=0)
        else:
            word_emb = self.word_emb(word_ids)
        if self.to_norm:
            word_emb = F.normalize(word_emb, p=2, dim=-1)
        return self.dropout(word_emb)


class CharacterEmbedding(nn.Module):
    def __init__(self, num_chars, char_dim, drop_rate, to_norm=False):
        super(CharacterEmbedding, self).__init__()
        self.char_emb = nn.Embedding(num_chars, char_dim, padding_idx=0)
        kernels, channels = [1, 2, 3, 4], [10, 20, 30, 40]
        self.char_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=char_dim, out_channels=channel, kernel_size=(1, kernel), stride=(1, 1), padding=0,
                          bias=True),
                nn.ReLU()
            ) for kernel, channel in zip(kernels, channels)
        ])
        self.dropout = nn.Dropout(p=drop_rate)
        self.to_norm = to_norm

    def forward(self, char_ids):
        char_emb = self.char_emb(char_ids)  # (batch_size, w_seq_len, c_seq_len, char_dim)
        if self.to_norm:
            char_emb = F.normalize(char_emb, p=2, dim=-1)
        char_emb = self.dropout(char_emb)
        char_emb = char_emb.permute(0, 3, 1, 2)  # (batch_size, char_dim, w_seq_len, c_seq_len)
        char_outputs = []
        for conv_layer in self.char_convs:
            output = conv_layer(char_emb)
            output, _ = torch.max(output, dim=3, keepdim=False)  # reduce max (batch_size, channel, w_seq_len)
            char_outputs.append(output)
        char_output = torch.cat(char_outputs, dim=1)  # (batch_size, sum(channels), w_seq_len)
        return char_output.permute(0, 2, 1)  # (batch_size, w_seq_len, sum(channels))


class Embedding(nn.Module):
    def __init__(self, num_words, num_chars, word_dim, char_dim, drop_rate, 
                    out_dim, word_vectors=None, to_norm=False, fq='word+text'):
        super(Embedding, self).__init__()
        self.fq = fq.split('+')
        assert len(self.fq) > 0 and all(item in ['word', 'text'] for item in self.fq)
        self.word_emb = WordEmbedding(num_words, word_dim, drop_rate, to_norm, word_vectors=word_vectors)
        self.char_emb = CharacterEmbedding(num_chars, char_dim, drop_rate, to_norm)
        # output linear layer
        in_dim = word_dim if 'word' in self.fq else 0
        in_dim += 100 if 'char' in self.fq else 0
        self.linear = Conv1D(in_dim=in_dim, out_dim=out_dim, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, word_ids, char_ids):
        word_emb = self.word_emb(word_ids)  # (batch_size, w_seq_len, word_dim)
        char_emb = self.char_emb(char_ids)  # (batch_size, w_seq_len, 100)
        emb = []
        if 'word' in self.fq: emb.append(word_emb)
        if 'char' in self.fq: emb.append(char_emb)
        if len(emb) > 1: emb = torch.cat(emb, dim=2)  # (batch_size, w_seq_len, word_dim + 100)
        else: emb = emb[0]
        emb = self.linear(emb)  # (batch_size, w_seq_len, dim)
        return emb


class VisualProjection(nn.Module):
    def __init__(self, visual_dim, dim, drop_rate=0.0):
        super(VisualProjection, self).__init__()
        self.drop = nn.Dropout(p=drop_rate)
        self.linear = Conv1D(in_dim=visual_dim, out_dim=dim, kernel_size=1, stride=1, bias=True, padding=0)

    def forward(self, visual_features):
        # the input visual feature with shape (batch_size, seq_len, visual_dim)
        visual_features = self.drop(visual_features)
        output = self.linear(visual_features)  # (batch_size, seq_len, dim)
        return output


class HighLightLayer(nn.Module):
    def __init__(self, dim):
        super(HighLightLayer, self).__init__()
        self.conv1d = Conv1D(in_dim=dim, out_dim=1, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x, mask):
        # compute logits
        logits = self.conv1d(x)
        logits = logits.squeeze(2)
        logits = mask_logits(logits, mask)
        # compute score
        scores = nn.Sigmoid()(logits)
        return scores

    @staticmethod
    def compute_loss(scores, labels, mask, epsilon=1e-12):
        labels = labels.type(torch.float32)
        weights = torch.where(labels == 0.0, labels + 1.0, 2.0 * labels)
        loss_per_location = nn.BCELoss(reduction='none')(scores, labels)
        loss_per_location = loss_per_location * weights
        mask = mask.type(torch.float32)
        loss = torch.sum(loss_per_location * mask) / (torch.sum(mask) + epsilon)
        return loss


class ConditionedPredictor(nn.Module):
    def __init__(self, dim, predictor):
        super(ConditionedPredictor, self).__init__()
        self.start_encoder = predictor
        self.end_encoder = copy.deepcopy(self.start_encoder)
        self.start_block = nn.Sequential(
            Conv1D(in_dim=2 * dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(),
            Conv1D(in_dim=dim, out_dim=1, kernel_size=1, stride=1, padding=0, bias=True)
        )
        self.end_block = nn.Sequential(
            Conv1D(in_dim=2 * dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(),
            Conv1D(in_dim=dim, out_dim=1, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x, mask):
        # encode
        start_features = self.start_encoder(target=x, tmask=mask)  # (batch_size, seq_len, dim)
        end_features = self.end_encoder(target=start_features, tmask=mask)
        # predict
        start_logits = self.start_block(torch.cat([start_features, x], dim=2))  # (batch_size, seq_len, 1)
        end_logits = self.end_block(torch.cat([end_features, x], dim=2))
        start_logits = mask_logits(start_logits.squeeze(2), mask=mask)
        end_logits = mask_logits(end_logits.squeeze(2), mask=mask)
        return start_logits, end_logits

    @staticmethod
    def extract_index(start_logits, end_logits):
        start_prob = nn.Softmax(dim=1)(start_logits)
        end_prob = nn.Softmax(dim=1)(end_logits)
        outer = torch.matmul(start_prob.unsqueeze(dim=2), end_prob.unsqueeze(dim=1))
        outer = torch.triu(outer, diagonal=0)
        _, start_index = torch.max(torch.max(outer, dim=2)[0], dim=1)  # (batch_size, )
        _, end_index = torch.max(torch.max(outer, dim=1)[0], dim=1)  # (batch_size, )
        return start_index, end_index

    @staticmethod
    def compute_cross_entropy_loss(start_logits, end_logits, start_labels, end_labels):
        start_loss = nn.CrossEntropyLoss(reduction='mean')(start_logits, start_labels)
        end_loss = nn.CrossEntropyLoss(reduction='mean')(end_logits, end_labels)
        return start_loss + end_loss
