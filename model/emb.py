import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.layers import Embedding, VisualProjection, downscale1d
from model.layers import ConditionedPredictor, HighLightLayer
from model.encoder import GuidedAttentionEncoder
from model.predictor import LSTMBlock
from model.fuser import VSLFuser
from model.position import CosinePosition
from model.proposal import TAN2dProposal
from model.ranker import Conv2dRanker
from transformers import AdamW, get_linear_schedule_with_warmup

from util.runner_utils import calculate_batch_iou


def build_optimizer_and_scheduler(model, configs):
    no_decay = ['bias', 'layer_norm', 'LayerNorm']  # no decay for parameters of layer norm and bias
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=configs.init_lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, configs.num_train_steps * configs.warmup_proportion,
                                                configs.num_train_steps)
    return optimizer, scheduler


class EMB(nn.Module):
    def __init__(self, configs, word_vectors):
        super(EMB, self).__init__()
        self.configs = configs
        self.embedding_net = Embedding(num_words=configs.word_size, num_chars=configs.char_size, out_dim=configs.dim,
                                       word_dim=configs.word_dim, char_dim=configs.char_dim, word_vectors=word_vectors,
                                       drop_rate=configs.drop_rate)
        self.video_affine = VisualProjection(visual_dim=configs.video_feature_dim, dim=configs.dim,
                                             drop_rate=configs.drop_rate)
        # positional encoding
        self.position = CosinePosition(**vars(configs))
        # encoder
        params = dict(**vars(configs))
        params['proposal'] = TAN2dProposal(**vars(configs))
        self.encoder = GuidedAttentionEncoder(**params)
        # fuser
        self.fuser1 = VSLFuser(**vars(configs))
        self.fuser2 = VSLFuser(**vars(configs))
        # query-guided highlighting
        self.highlight_layer = HighLightLayer(dim=configs.dim)
        # conditioned predictor
        params = dict(**vars(configs))
        params['num_layers'], params['drop_rate'] = 1, 0
        self.predictor = ConditionedPredictor(configs.dim, LSTMBlock(**params))
        # ranker
        self.ranker = Conv2dRanker(**vars(configs))
        # init parameters
        self.init_parameters()

    def init_parameters(self):
        def init_weights(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                m.reset_parameters()
        self.apply(init_weights)

    def forward(self, word_ids, char_ids, video_features, v_mask, q_mask, moments=None):
        video_features = self.video_affine(video_features)
        query_features = self.embedding_net(word_ids, char_ids)
        video_features, query_features = self.position(video_features), self.position(query_features)
        ((video_features, vquery_features), 
         (proposal_features, pquery_features), 
         proposals, p_mask) = self.encoder(vfeats=video_features, 
                            qfeats=query_features, vmask=v_mask, qmask=q_mask,
                            moments=moments)
        # conduct frame-level regression
        features = self.fuser1(vfeats=video_features, qfeats=vquery_features,
                            vmask=v_mask, qmask=q_mask)
        h_score = self.highlight_layer(features, v_mask)
        features = features * h_score.unsqueeze(2)
        start_logits, end_logits = self.predictor(features, mask=v_mask)
        # conduct proposal level ranking
        features = self.fuser2(vfeats=proposal_features, qfeats=pquery_features,
                            vmask=p_mask, qmask=q_mask)
        # convert proposal's feature and mask back to 2d for ranking
        p_score = self.ranker(features, p_mask)
        # flatten proposal's bounds, mask and scores as outputs
        return h_score, start_logits, end_logits, proposals, p_score, p_mask

    def compute_ranking_loss(self, moments, proposals, scores, mask, min_iou, max_iou):
        return self.ranker.compute_loss(
            moments=moments, bounds=proposals, 
            scores=scores, mask=mask, min_iou=min_iou, max_iou=max_iou)

    def compute_highlight_loss(self, scores, labels, mask, moments, 
                    proposals, pscores, pmask, threshold=0.5, extend=0.1):
        # get the most cconfident proposals as the pseudo labels
        pseudo = self.ranker.topk_confident(torch.stack([proposals[...,0], proposals[...,1]+1], dim=-1), 
            pscores, pmask, torch.stack([moments[...,0], moments[...,1]+1], dim=-1), threshold, k=1).squeeze(1)
        # start/end: (<bsz>)
        start = torch.min(moments[:,0], pseudo[:,0])
        end = torch.min(moments[:,1], pseudo[:,1])
        # extend the pseudo boundary: start/end (<bsz>)
        extension = ((end - start + 1).float() * extend).round().long()
        start = (start - extension).clamp(min=0)
        end = torch.min(mask.sum(-1).long() - 1, end + extension)
        # build the frame-wise binary labels: (<bsz>, <#frames>)
        labels = torch.arange(mask.shape[1], device=scores.device).view(1, -1)
        labels = (labels >= start.view(-1, 1)) * (labels <= end.view(-1, 1))
        labels = labels.detach().long()
        return self.highlight_layer.compute_loss(scores=scores, labels=labels, mask=mask)

    def compute_loss(self, start_logits, end_logits, start_labels, end_labels,
            proposals, pscores, pmask, threshold=0.5):
        moments = torch.stack([start_labels, end_labels], dim=1)
        # get the most confident proposals as the pseudo labels
        pseudo = self.ranker.topk_confident(torch.stack([proposals[...,0], proposals[...,1]+1], dim=-1), 
            pscores, pmask, torch.stack([moments[...,0], moments[...,1]+1], dim=-1), threshold, k=1).squeeze(1)
        # construct labels for the start and end logits
        start = torch.arange(start_logits.shape[1], device=start_logits.device).view(1, -1)
        start = ((start >= torch.min(moments[:,0], pseudo[:,0]).view(-1, 1)) *
                 (start <= torch.max(moments[:,0], pseudo[:,0]).view(-1, 1)))
        end = torch.arange(end_logits.shape[1], device=end_logits.device).view(1, -1)
        end = ((end >= torch.min(moments[:,1], pseudo[:,1]).view(-1, 1)) *
               (end <= torch.max(moments[:,1], pseudo[:,1]).view(-1, 1)))
        # maximise the logit of frames falling in the candidated spans
        start_loss = -(F.softmax(start_logits, dim=-1) * start).sum(-1).log().mean()
        end_loss = -(F.softmax(end_logits, dim=-1) * end).sum(-1).log().mean()
        return start_loss + end_loss

    def extract_index(self, start_logits, end_logits, proposals, pscores, pmask):
        # get the most confident proposals as the pseudo labels
        ppseudo = self.ranker.topk_confident(torch.stack([proposals[...,0], proposals[...,1]+1], dim=-1), 
            pscores, pmask, None, 0., k=1).squeeze(1)
        bpseudo = torch.stack(self.predictor.extract_index(start_logits, end_logits), dim=-1)
        # construct labels for the start and end logits
        start = torch.arange(start_logits.shape[1], device=start_logits.device).view(1, -1)
        start = ((start >= torch.min(bpseudo[:,0], ppseudo[:,0]).view(-1, 1)) *
                 (start <= torch.max(bpseudo[:,0], ppseudo[:,0]).view(-1, 1)))
        end = torch.arange(end_logits.shape[1], device=end_logits.device).view(1, -1)
        end = ((end >= torch.min(bpseudo[:,1], ppseudo[:,1]).view(-1, 1)) *
               (end <= torch.max(bpseudo[:,1], ppseudo[:,1]).view(-1, 1)))
        assert (start.sum(-1) >= 1).all() and (end.sum(-1) >= 1).all()
        return start, end

