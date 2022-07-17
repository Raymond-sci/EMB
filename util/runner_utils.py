import os
import re
import glob
import random
import shutil
import zipfile
import logging
import multiprocessing as mp
import numpy as np
from contextlib import contextmanager

import torch
import torch.utils.data
import torch.backends.cudnn
from tqdm import tqdm
from util.data_util import index_to_time


def do_nothing(*args, **kwargs):
    pass


def set_th_config(seed):
    if not seed: return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def save_checkpoint(path, state, is_best, filename='checkpoint.pth.tar'):
    fname = os.path.join(path, filename)
    torch.save(state, fname)
    bname = os.path.join(path, 'model_best.pth.tar')
    if is_best: shutil.copyfile(fname, bname)


def load_checkpoint(path):
    return os.path.join(path, 'model_best.pth.tar')


def convert_length_to_mask(lengths, max_len=None):
    max_len = lengths.max().item() if max_len is None else max_len
    mask = torch.arange(max_len, device=lengths.device).expand(lengths.size()[0], max_len) < lengths.unsqueeze(1)
    mask = mask.float()
    return mask


def calculate_iou_accuracy(ious, threshold):
    total_size = float(len(ious))
    count = 0
    for iou in ious:
        if iou >= threshold:
            count += 1
    return float(count) / total_size * 100.0


def calculate_iou(i0, i1):
    union = (min(i0[0], i1[0]), max(i0[1], i1[1]))
    inter = (max(i0[0], i1[0]), min(i0[1], i1[1]))
    iou = 1.0 * (inter[1] - inter[0]) / (union[1] - union[0])
    return max(0.0, iou)


def calculate_batch_iou(i0, i1):
    assert i0.dim() == i1.dim()
    union = (torch.min(i0[...,0], i1[...,0]), torch.max(i0[...,1], i1[...,1]))
    inter = (torch.max(i0[...,0], i1[...,0]), torch.min(i0[...,1], i1[...,1]))
    iou = 1. * (inter[1] - inter[0]).clamp(min=0) / (union[1] - union[0]).clamp(min=1e-5)
    return iou

@torch.no_grad()
def eval_test(model, data_loader, device, mode='test', epoch=None, global_step=None, elastic=False):
    ious = []
    for idx, data in tqdm(enumerate(data_loader), total=len(data_loader), desc='evaluate {}'.format(mode), leave=False):
        # prepare features
        records, vfeats, vfeat_lens, word_ids, char_ids, s_labels, e_labels, _ = data
        vfeats, vfeat_lens = vfeats.to(device), vfeat_lens.to(device)
        word_ids, char_ids = word_ids.to(device), char_ids.to(device)
        s_labels, e_labels = s_labels.to(device), e_labels.to(device)
        # generate mask
        query_mask = (torch.zeros_like(word_ids) != word_ids).float().to(device)
        video_mask = convert_length_to_mask(vfeat_lens).to(device)
        # compute predicted results
        (_, start_logits, end_logits, 
         proposals, p_score, p_mask) = model(word_ids, char_ids, vfeats, video_mask, query_mask)
        if not elastic:
            # evaluate the bounding branch
            start_indices, end_indices = model.predictor.extract_index(start_logits, end_logits)
        else:
            # evaluate the elastic boundary
            start_mask, end_mask = model.extract_index(start_logits, end_logits, proposals, p_score, p_mask)
            start_indices, end_indices = convert_ela_to_det(start_mask, end_mask, s_labels, e_labels, video_mask)
        start_indices = start_indices.cpu().numpy()
        end_indices = end_indices.cpu().numpy()
        for record, start_index, end_index in zip(records, start_indices, end_indices):
            start_time, end_time = index_to_time(start_index, end_index, record["v_len"], record["duration"])
            iou = calculate_iou(i0=[start_time, end_time], i1=[record["s_time"], record["e_time"]])
            ious.append(iou)
    r1i3 = calculate_iou_accuracy(ious, threshold=0.3)
    r1i5 = calculate_iou_accuracy(ious, threshold=0.5)
    r1i7 = calculate_iou_accuracy(ious, threshold=0.7)
    mi = np.mean(ious) * 100.0
    # write the scores
    score_str = "Epoch {}, Step {}:\t".format(epoch, global_step)
    score_str += "Rank@1, IoU=0.3: {:.2f}\t".format(r1i3)
    score_str += "Rank@1, IoU=0.5: {:.2f}\t".format(r1i5)
    score_str += "Rank@1, IoU=0.7: {:.2f}\t".format(r1i7)
    score_str += "mean IoU: {:.2f}".format(mi)
    return r1i3, r1i5, r1i7, mi, score_str


def batch_index_to_time(start_index, end_index, num_units, duration):
    assert all(start_index.dim() == item.dim() for item in [end_index, num_units, duration])
    assert (start_index < num_units).all() and (end_index < num_units).all()
    start_time = 1. * start_index / num_units * duration
    end_time = 1. * (end_index + 1) / num_units * duration
    return start_time, end_time


def convert_ela_to_det(start, end, start_labels, end_labels, vmask):
    """
    the multi-candidates boundaries is converted here
    by selecting the candidate with the largest overlap with the ground-truth boundary.
    This is equivalent to 
    considering an elastic boundary (multi-candidates) is correctly predicted
    when any of the candidate's overlap to the ground-truth is greater than a threshold

    Such implementation is to 
    fit in the standard protocol for evaluating determined boundary,
    """
    # enumerate all possible boundary
    lens = vmask.sum(-1).view(-1, 1).long()
    cands = (start.unsqueeze(-1) * end.unsqueeze(1)).float() # Nx128x128
    cands *= vmask.unsqueeze(-1) * vmask.unsqueeze(1)
    cands = torch.triu(cands, diagonal=0)
    cands_start = torch.arange(start.shape[1], device=start.device)
    cands_start = torch.min(cands_start.view(1, -1), lens-1).unsqueeze(-1)
    cands_end = torch.arange(end.shape[1], device=end.device)
    cands_end = torch.min(cands_end.view(1, -1), lens-1).unsqueeze(1) + 1
    cands_bound = torch.stack([cands_start.repeat(1, 1, end.shape[1]), 
                                cands_end.repeat(1, start.shape[1], 1)], 
                                dim=-1).view(cands.shape[0], -1, 2) # Nx(128x128)x2
    # ious between all 
    moments = torch.stack([start_labels, end_labels+1], dim=1)
    ious = calculate_batch_iou(cands_bound, moments.unsqueeze(1)).view_as(cands) # Nx128x128
    # ious = torch.rand_like(ious)
    ious = ious * cands - (1 - cands) * 1e10
    # get cands with greatest iou with the ground-truth
    _, start_index = torch.max(torch.max(ious, dim=2)[0], dim=1)  # (batch_size, )
    _, end_index = torch.max(torch.max(ious, dim=1)[0], dim=1)  # (batch_size, )
    assert (start_index <= end_index).all()
    return start_index, end_index


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':.3f'):
        self.key = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{key} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class AverageMeters:
    """wrapper of average meters

    take key value pairs as inputs to update meter
    if unknown keys given, then auto create meter
    """
    def __init__(self, **params):
        self.meters = {}
        self.params = params

    def __getattr__(self, key):
        if key not in ['val', 'sum', 'count', 'avg']:
            raise AttributeError('AverageMeters has no attribute "{}"'.format(key))
        return {name:getattr(meter, key) for name,meter in self.meters.items()}

    def update(self, n=1, **kwargs):
        for key, val in kwargs.items():
            if key not in self.meters:
                self.meters[key] = AverageMeter(key, **self.params)
            _val = val.item() if torch.is_tensor(val) else val
            self.meters[key].update(_val, n=n)

    def __str__(self):
        return '\t'.join([str(meter) for _, meter in self.meters.items()])


class ProgressMeters:
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def update(self, *args, **kwargs):
        self.meters.update(*args, **kwargs)

    def display(self, batch, logger=None):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(self.meters)]
        # entries += [str(meter) for meter in self.meters]
        if logger is not None: logger.info('\t'.join(entries))
        else: print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class TqdmHandler(logging.Handler):
    """tqdm logging handler
    
    use this logger instead of the StreamHandler to interact with tqdm progress bar
    """
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)


def get_logger(name=__name__, level=logging.DEBUG, to_file=None,
    fmt='[%(levelname)s][%(asctime)s]:\t%(message)s'):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if len(logger.handlers) < 2:
        handlers = [TqdmHandler()]
        if to_file and to_file != '':
            handlers.append(logging.FileHandler(to_file))

        for handler in handlers[len(logger.handlers):]:
            logger.addHandler(handler)

    for hidx, handler in enumerate(logger.handlers):
        logger.handlers[hidx].setLevel(level)
        logger.handlers[hidx].setFormatter(logging.Formatter(fmt))

    return logger