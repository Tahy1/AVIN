import logging
import os
import os.path as osp
import numpy as np
from sklearn.metrics import auc
import torch.nn.functional as F
import torch

class Averager:
    def __init__(self):
        self.epoch_loss = 0.0
        self.epoch_correct = 0
        self.epoch_total = 0

        self.iter_loss = 0.0
        self.iter_correct = 0
        self.iter_total = 0

        self.iter_step = 0
        self.log_step = 0

    def update(self, loss, correct, total):
        self.epoch_loss += loss
        self.iter_loss += loss

        self.iter_correct += correct
        self.iter_total += total
        self.epoch_correct += correct
        self.epoch_total += total

        self.iter_step += 1
        self.log_step += 1

    def reset(self, type):
        assert type in ['epoch', 'iter']
        if type == 'iter':
            self.iter_loss = 0.0
            self.iter_correct = 0
            self.iter_total = 0
            self.log_step = 0 # every N steps to calc avg and print log
        elif type == 'epoch':
            self.epoch_loss = 0.0
            self.epoch_correct = 0
            self.epoch_total = 0
            self.iter_step = 0

    def reset_all(self):
        self.epoch_loss = 0.0
        self.epoch_correct = 0
        self.epoch_total = 0

        self.iter_loss = 0.0
        self.iter_correct = 0
        self.iter_total = 0

        self.iter_step = 0
        self.log_step = 0

    def average(self, type):
        assert type in ['epoch', 'iter']
        if type == 'iter':
            return self.iter_loss / self.log_step, self.iter_correct / self.iter_total
        elif type == 'epoch':
            return self.epoch_loss / self.iter_step, self.epoch_correct / self.epoch_total

def get_root_logger(log_file, name, log_level=logging.INFO, file_mode='w'):
    logger = logging.getLogger(name)

    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    file_handler = logging.FileHandler(log_file, file_mode)
    handlers.append(file_handler)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)
    logger.setLevel(log_level)

    return logger

def mkdir_or_exist(dir_name, mode=0o777):
    if dir_name == '':
        return
    dir_name = osp.expanduser(dir_name)
    os.makedirs(dir_name, mode, exist_ok=True)

def metric(pred_maps, gt_maps, thres, retmap=False):
    predmaps = []
    B, H, W = pred_maps.shape
    pred_maps = F.interpolate(pred_maps.unsqueeze(1), (224, 224), mode='bilinear', align_corners=False).squeeze() # B H W
    pred_maps = normalize_img(pred_maps)
    infer_maps = torch.zeros_like(pred_maps)
    infer_maps[pred_maps >= thres] = 1
    inters = torch.sum(infer_maps * gt_maps, dim=(1,2)) # B
    unions = torch.sum(gt_maps, dim=(1,2)) + torch.sum(infer_maps * (gt_maps == 0), dim=(1,2)) # B
    cious = inters / unions # B
    gtnsas = torch.zeros_like(gt_maps) # B H W
    gtnsas[gt_maps==0] = 1 # B H W
    occupys = torch.sum(infer_maps * gtnsas, dim=(1,2)) # B
    totals = torch.sum(gtnsas, dim=(1,2)) # B
    nsas = 1 - occupys / totals # B
    ciou = torch.mean((cious >= 0.5) + 0.)
    nsa = torch.mean(nsas[~torch.isnan(nsas)])
    i = torch.arange(21).unsqueeze(0).repeat(B, 1)
    results = torch.mean((cious.unsqueeze(-1) >= 0.05 * i) + 0., dim=0).cpu()
    x = [0.05 * i for i in range(21)]
    auc_ = auc(x, results)
    ret = [ciou, auc_, nsa]
    if retmap:
        predmaps = np.stack(predmaps)
        ret.append(predmaps)
    return ret

def normalize_img(value):
    B = value.shape[0]
    vmin = torch.min(value.view(B, -1), dim=1)[0].unsqueeze(-1).unsqueeze(-1)
    vmax = torch.max(value.view(B, -1), dim=1)[0].unsqueeze(-1).unsqueeze(-1)
    value = (value - vmin) / (vmax - vmin)
    return value
    