import os
import copy
import time
import argparse

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from datetime import datetime
from functools import partialmethod

from model.emb import EMB, build_optimizer_and_scheduler
from model.threshold import SigmoidThreshold
from util.data_util import load_video_features, save_json, load_json
from util.data_gen import gen_or_load_dataset
from util.data_loader import get_train_loader, get_test_loader
from util.runner_utils import set_th_config, convert_length_to_mask, eval_test, \
    ProgressMeters, AverageMeters, batch_index_to_time, calculate_batch_iou, \
    get_logger, save_checkpoint, load_checkpoint

parser = argparse.ArgumentParser()
# data parameters
parser.add_argument('--save_dir', type=str, default='datasets', help='path to save processed dataset')
parser.add_argument('--task', type=str, default='charades', help='target task')
parser.add_argument('--fv', type=str, default='new', help='[new | org] for visual features')
parser.add_argument('--fq', type=str, default='word+char', help='query features: word, char, word+char')
parser.add_argument('--max_pos_len', type=int, default=128, help='maximal position sequence length allowed')
# model parameters
parser.add_argument("--word_size", type=int, default=None, help="number of words")
parser.add_argument("--char_size", type=int, default=None, help="number of characters")
parser.add_argument("--word_dim", type=int, default=300, help="word embedding dimension")
parser.add_argument("--video_feature_dim", type=int, default=1024, help="video feature input dimension")
parser.add_argument("--char_dim", type=int, default=50, help="character dimension, set to 100 for activitynet")
parser.add_argument("--dim", type=int, default=128, help="hidden size")
parser.add_argument("--highlight_lambda", type=float, default=5.0, help="lambda for highlight region")
parser.add_argument("--rank_lambda", type=float, default=1.0, help="lambda for ranking branch")
parser.add_argument("--num_heads", type=int, default=8, help="number of heads")
parser.add_argument("--num_layers", type=int, default=1, help="number of convolution layers")
parser.add_argument("--drop_rate", type=float, default=0.2, help="dropout rate")
parser.add_argument("--min_iou", type=float, default=0.3, help="lower bound of iou")
parser.add_argument("--max_iou", type=float, default=0.7, help="upper bound of iou")
parser.add_argument("--threshold", type=float, default=0.5, help="lambda for ranking loss")
# training/evaluation parameters
parser.add_argument("--gpu_idx", type=str, default="0", help="GPU index")
parser.add_argument("--seed", type=int, default=None, help="random seed")
parser.add_argument("--mode", type=str, default="train", help="[train | test]")
parser.add_argument("--elastic", action='store_true', help='if test the model according to elastic boundary')
parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
parser.add_argument("--batch_size", type=int, default=16, help="batch size")
parser.add_argument("--num_train_steps", type=int, default=None, help="number of training steps")
parser.add_argument("--init_lr", type=float, default=0.0005, help="initial learning rate")
parser.add_argument("--clip_norm", type=float, default=1.0, help="gradient clip norm")
parser.add_argument("--warmup_proportion", type=float, default=0.0, help="warmup proportion")
parser.add_argument("--extend", type=float, default=0.1, help="highlight region extension")
parser.add_argument("--period", type=int, default=5, help="training loss print period")
parser.add_argument('--deploy', action='store_true', help='if to save checkpoint and logs')
parser.add_argument('--model_dir', type=str, default='sessions', help='path to save trained model weights')
parser.add_argument('--model_name', type=str, default=datetime.now().strftime('%Y%m%d-%H%M%S'), help='model name')
parser.add_argument('--suffix', type=str, default=None, help='set to the last `_xxx` in ckpt repo to eval results')
gconfigs = parser.parse_args()


def setup(configs):
    # set tensorflow configs
    set_th_config(configs.seed)

    # prepare or load dataset
    dataset = gen_or_load_dataset(configs)
    configs.char_size = dataset['n_chars']
    configs.word_size = dataset['n_words']

    # get train and test loader
    visual_features = load_video_features(os.path.join('data', 'features', configs.task, configs.fv), configs.max_pos_len)
    train_loader = get_train_loader(dataset=dataset['train_set'], video_features=visual_features, configs=configs)
    val_loader = None if dataset['val_set'] is None else get_test_loader(dataset['val_set'], visual_features, configs)
    test_loader = get_test_loader(dataset=dataset['test_set'], video_features=visual_features, configs=configs)
    configs.num_train_steps = len(train_loader) * configs.epochs
    num_train_batches = len(train_loader)
    num_val_batches = 0 if val_loader is None else len(val_loader)
    num_test_batches = len(test_loader)

    # Device configuration
    cuda_str = 'cuda' if configs.gpu_idx is None else 'cuda:{}'.format(configs.gpu_idx)
    device = torch.device(cuda_str if torch.cuda.is_available() else 'cpu')

    # create model dir
    model_dir = os.path.join(configs.model_dir, configs.task, configs.model_name)
    # save snapshot
    if configs.deploy: os.makedirs(model_dir, exist_ok=True)
    # get logger, log to file if deploy
    logger = get_logger(level=('DEBUG' if not configs.deploy else 'INFO'),
                        to_file=(None if not configs.deploy else os.path.join(model_dir, 'log')))

    return dataset, train_loader, val_loader, test_loader, num_train_batches, num_val_batches, num_test_batches, device, model_dir, logger


# training
def train(configs):
    (dataset, train_loader, val_loader, test_loader, num_train_batches, 
     num_val_batches, num_test_batches, device, model_dir, logger) = setup(configs)
    if configs.deploy:
        save_json(vars(configs), os.path.join(model_dir, 'configs.json'), sort_keys=True, save_pretty=True)
    # build model
    model = EMB(configs=configs, word_vectors=dataset['word_vector']).to(device)
    optimizer, scheduler = build_optimizer_and_scheduler(model, configs=configs)
    threshold = SigmoidThreshold(start=0, end=configs.epochs, low=configs.threshold)
    # start training
    log_period = num_train_batches // configs.period
    eval_period = num_train_batches // 2
    best_results = (-1., -1., -1., -1)
    logger.debug('Start training...')
    global_step = 0
    for epoch in tqdm(range(configs.epochs), desc="Overall", leave=True):
        meters = ProgressMeters(len(train_loader), AverageMeters(), 
                                prefix="Epoch: [%3d]" % (epoch + 1))
        model.train()
        for local_step, data in enumerate(tqdm(train_loader, total=num_train_batches, 
                                        desc='Epoch %3d' % (epoch + 1), leave=False)):
            global_step += 1
            records, vfeats, vfeat_lens, word_ids, char_ids, s_labels, e_labels, h_labels = data
            # prepare metd
            durations = [record['duration'] for record in records]
            durations = torch.as_tensor(durations).float().to(device)
            # prepare features
            vfeats, vfeat_lens = vfeats.to(device), vfeat_lens.to(device)
            word_ids, char_ids = word_ids.to(device), char_ids.to(device)
            s_labels, e_labels, h_labels = s_labels.to(device), e_labels.to(device), h_labels.to(device)
            # generate mask
            query_mask = (torch.zeros_like(word_ids) != word_ids).float().to(device)
            video_mask = convert_length_to_mask(vfeat_lens).to(device)
            # compute logits
            (h_score, start_logits, end_logits, 
             proposals, p_score, p_mask) = model(word_ids, char_ids, vfeats, video_mask, query_mask)
            # compute loss
            highlight_loss = model.compute_highlight_loss(h_score, h_labels, video_mask,
                torch.stack([s_labels, e_labels], dim=-1), proposals, p_score, p_mask,
                threshold=threshold(epoch, reverse=True), extend=configs.extend)
            loc_loss = model.compute_loss(start_logits, end_logits, s_labels, 
                e_labels, proposals, p_score, p_mask,
                threshold=threshold(epoch, reverse=True))
            # convert the index label of ground-truth moments and proposals to time labels
            moments = batch_index_to_time(s_labels, e_labels, vfeat_lens, durations)
            moments = torch.stack(moments, dim=-1)
            p_time = batch_index_to_time(proposals[...,0], proposals[...,1], 
                vfeat_lens.unsqueeze(-1), durations.unsqueeze(-1))
            p_time = torch.stack(p_time, dim=-1)
            p_time *= p_mask.unsqueeze(-1)
            rank_loss = model.compute_ranking_loss(moments, p_time, p_score, 
                        p_mask, min_iou=configs.min_iou, max_iou=configs.max_iou)
            total_loss = (loc_loss + 
                          configs.highlight_lambda * highlight_loss +
                          configs.rank_lambda * rank_loss)
            # compute and apply gradients
            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), configs.clip_norm)  # clip gradient
            optimizer.step()
            scheduler.step()
            # meters.update(Threshold=threshold(epoch, reverse=True), 
            meters.update(High=highlight_loss, Rank=rank_loss, Loc=loc_loss, 
                          Loss=total_loss)
            if local_step % log_period == 0:
                meters.display(local_step, logger)
            # evaluate
            if global_step % eval_period == 0 or global_step % num_train_batches == 0:
                model.eval()
                r1i3, r1i5, r1i7, mi, score_str = eval_test(model=model, 
                    data_loader=test_loader, device=device,
                    mode='test', epoch=epoch + 1, global_step=global_step,
                    elastic=configs.elastic)
                logger.info('Epoch: %3d | Step: %5d | r1i3: %.2f | r1i5: %.2f | r1i7: %.2f | mIoU: %.2f' % (
                    epoch + 1, global_step, r1i3, r1i5, r1i7, mi))
                # TODO: don't save checkpoint files
                results = (r1i3, r1i5, r1i7, mi)
                if configs.deploy:
                    save_checkpoint(model_dir, {
                            'epoch': epoch + 1,
                            'step': global_step,
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                        }, is_best=(sum(results) > sum(best_results)))
                if sum(results) > sum(best_results): best_results = results
                model.train()
    logger.debug('Done training')
    logger.info('Best results yielded - r1i3: %.2f | r1i5: %.2f | r1i7: %.2f | mIoU: %.2f' % best_results)
    return best_results


def test(configs):
    # setup
    (dataset, train_loader, val_loader, test_loader, num_train_batches, 
     num_val_batches, num_test_batches, device, model_dir, logger) = setup(configs)
    if not os.path.exists(model_dir):
        raise ValueError('No pre-trained weights exist')
    # load previous configs
    pre_configs = load_json(os.path.join(model_dir, "configs.json"))
    parser.set_defaults(**pre_configs)
    configs = parser.parse_args()
    # build model
    model = EMB(configs=configs, word_vectors=dataset['word_vector']).to(device)
    # get last checkpoint file
    filename = load_checkpoint(model_dir)
    model.load_state_dict(torch.load(filename)['state_dict'])
    model.eval()
    r1i3, r1i5, r1i7, mi, _ = eval_test(model=model, data_loader=test_loader, 
        device=device, mode='test', elastic=configs.elastic)
    logger.info("\x1b[1;31m" + "Rank@1, IoU=0.3:\t{:.2f}".format(r1i3) + "\x1b[0m")
    logger.info("\x1b[1;31m" + "Rank@1, IoU=0.5:\t{:.2f}".format(r1i5) + "\x1b[0m")
    logger.info("\x1b[1;31m" + "Rank@1, IoU=0.7:\t{:.2f}".format(r1i7) + "\x1b[0m")
    logger.info("\x1b[1;31m" + "{}:\t{:.2f}".format("mean IoU".ljust(15), mi) + "\x1b[0m")
    return r1i3, r1i5, r1i7, mi


if __name__ == '__main__':
    if gconfigs.mode.lower() == 'train':
        train(gconfigs)
    elif gconfigs.mode.lower() == 'test':
        test(gconfigs)
    else:
        raise NotImplementedError('mode should be one of ("train", "test")')
