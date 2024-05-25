import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage import measure
import numpy as np
import math

from .original_loss import DiceLoss, BCELoss, MSELoss, MAELoss

def KthBit(n, k):
    k = int(k)
    if type(n) == torch.Tensor:
        n = n.int()
        bitmask = 1 << (k - 1)
        # Use bitwise AND to check if the k-th bit is set
        result = (n & bitmask) >> (k - 1)
    elif type(n) == np.ndarray:
        n = n.astype(int)
        bitmask = 1 << (k - 1)
        # Use bitwise AND to check if the k-th bit is set
        result = (n & bitmask) >> (k - 1)
    else:
        result = (n & (1 << (k - 1))) >> (k - 1)

    return result


class DSLoss:
    def __init__(self, criterion):
        self.criterion = criterion

    def __call__(self, inputs, targets):
        if isinstance(targets, tuple):
            targets = targets[0]
        channels = inputs.shape[0]
        total_loss = torch.tensor(0)
        for loss_func in self.criterion:
            for idx in range(channels):
                loss = loss_func(inputs[idx, :, :], targets)
                total_loss = total_loss + loss

        return total_loss


# Size-Invariant Loss
class SILoss:
    def __init__(self, criterion):
        self.criterion = criterion
        self.DSLoss = DSLoss(criterion)

    def __call__(self, pred, gt, weight):
        b, h, w = gt.shape
        device = gt.device
        loss = torch.tensor(0.0, requires_grad=True).to(device)
        for batch in range(b):
            image_loss = torch.tensor(0.0, requires_grad=True).to(device)
            max_weight = torch.max(weight[batch])
            num_features = 0
            if max_weight > 0:
                num_features = int(math.log(torch.max(max_weight), 2)) + 1
            if num_features < 1:
                image_loss = self.DSLoss(pred[batch], gt[batch])
                loss = loss + image_loss
            else:
                non_saliency_mask = ((weight[batch]) == 0).to(device)
                non_saliency_pred_mask = non_saliency_mask * pred[batch].mean(dim=0) # pred_batch = pred[batch]
                non_saliency_gt_mask = non_saliency_mask * gt[batch]
                non_saliency_size = torch.sum(non_saliency_mask)
                # alpha
                non_saliency_times = non_saliency_size / (h * w - non_saliency_size)
                if non_saliency_size != 0:
                    non_saliency_loss = torch.sum((non_saliency_pred_mask - non_saliency_gt_mask) ** 2) / non_saliency_size
                    # non_saliency_loss = torch.sum((non_saliency_pred_mask) ** 2) / non_saliency_size
                else:
                    non_saliency_loss = torch.tensor(0.0).to(device)
                # loss in the background frame
                non_saliency_loss = non_saliency_loss * non_saliency_times

                for step in range(num_features + 1):
                    if step != 0:
                        mask_frame = (KthBit(weight[batch], step) == 1).int()
                        mask_frame_size = np.sum(mask_frame.numpy() > 0) # numpy is much faster!
                        # mask_frame_size = torch.sum(mask_frame > 0)

                        if mask_frame_size == 0:
                            continue

                        rows, cols = np.where(mask_frame > 0)
                        # rows, cols = torch.where(mask_frame > 0)
                        xmin, xmax = min(cols), max(cols)
                        ymin, ymax = min(rows), max(rows)

                        pred_mask = pred[batch, :, ymin:(ymax+1), xmin:(xmax+1)]
                        gt_mask = gt[batch, ymin:(ymax+1), xmin:(xmax+1)]

                        # loss in a object frame
                        ds_loss = self.DSLoss(pred_mask, gt_mask)
                        image_loss = image_loss + ds_loss

                loss = loss + image_loss + non_saliency_loss

        return loss / b
