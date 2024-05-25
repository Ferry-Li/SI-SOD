import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage import measure
import numpy as np
import math

EPS = 1e-5
# EDN Losses
def BCELoss(inputs, targets):
    bce = F.binary_cross_entropy(inputs, targets)
    return bce

def MSELoss(inputs, targets):
    _, _, h, w = inputs.shape
    size = h * w
    mse = torch.sum((inputs - targets) ** 2) / (size + EPS)
    return mse

def MAELoss(inputs, targets):
    _, _, h, w = inputs.shape
    size = h * w
    mae = torch.abs((inputs - targets)) / (size + EPS)
    return mae

def DiceLoss(inputs, targets):
    inter = (inputs * targets).sum()
    dice = (2 * inter + EPS) / (inputs.sum() + targets.sum() + EPS)
    return 1 - dice

class DSLoss:
    def __init__(self, criterion):
        self.criterion = criterion

    def __call__(self, inputs, targets, weight=None):
        if isinstance(targets, tuple):
            targets = targets[0]
        total_loss = torch.tensor(0)
        channels = inputs.shape[1]
        for loss_func in self.criterion:
            for idx in range(channels):
                loss = loss_func(inputs[:, idx, :, :], targets)
                total_loss = total_loss + loss

        return total_loss


# ICON Losses
# IoU Loss
def iou_loss(pred, mask):
    pred  = torch.sigmoid(pred)
    inter = (pred*mask).sum(dim=(0,1))
    union = (pred+mask).sum(dim=(0,1))
    iou  = 1-(inter+1)/(union-inter+1)
    return iou.mean()


# Structure Loss
def structure_loss(pred, mask):
    unsqueeze_mask = mask.unsqueeze(0).unsqueeze(0)
    weit  = 1+5*torch.abs(F.avg_pool2d(unsqueeze_mask, kernel_size=31, stride=1, padding=15)-unsqueeze_mask)
    weit  = weit.squeeze(0).squeeze(0)
    wbce  = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce  = (weit*wbce).sum(dim=(0,1))/weit.sum(dim=(0,1))

    pred  = torch.sigmoid(pred)
    inter = ((pred*mask)*weit).sum(dim=(0,1))
    union = ((pred+mask)*weit).sum(dim=(0,1))
    wiou  = 1-(inter+1)/(union-inter+1)
    return (wbce+wiou).mean()

# BCELogits
def BCELogits(pred, mask):
    return F.binary_cross_entropy_with_logits(pred, mask)