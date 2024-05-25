import torch
import numpy as np
import torch.nn.functional as F
import warnings
import math
from sklearn.metrics import roc_auc_score

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

def find_num_bin(k):
    n = 0
    while k != 0:
        k = k // 2
        n += 1
    return n


def mask_split_bit(pred, gt, split, step):
    # extract certain split part according to binary system
    step_mask = (KthBit(split, step) == 1)
    return pred * step_mask, gt * step_mask, step_mask

    # pred: masked pred, size = split_part * pred
    # gt: masked gt, size = split_part * gt
    # step_mask: masked image, size = split_part

def mask_split(pred, gt, split, step):
    # extract certain split part
    step_mask = (split == step)
    return pred * step_mask, gt * step_mask, step_mask

class Metric:
    def __init__(self):
        self.alpha = 0
    # single image eval functions
    # num should be consistent with transforms or image shape

    def eval_mae(self, pred, gt, split, scale=False):
        # size is considered in this metric!
        h, w = pred.shape
        eps = 1e-20
        if split is not None:
            # get the step: [1, 2, 3 ...] (0 is also possible!)
            # step_list = np.unique(split)
            if torch.max(split) == 0:
                max_step_list = 0
            else:
                max_step_list = math.floor(
                    math.log(torch.max(split), 2) + 1) + 1

            mae_list = list()
            non_saliency_times = 1
            is_alpha = False
            
            # breakpoint()
            for step in range(max_step_list):
                # extract certain split part
                if step != 0:
                    mask_pred, mask_gt, mask = mask_split_bit(
                        pred, gt, split, step)
                    if torch.sum(mask.int()) == 0:
                        continue
                    size_mask = torch.sum(mask)
                    mae_list.append(
                        torch.sum(torch.abs(mask_pred - mask_gt)) / size_mask)
                else:

                    mask_pred, mask_gt, mask = mask_split(
                        pred, gt, split, step)
                    if torch.sum(mask.int()) == 0:
                        continue
                    size_mask = torch.sum(mask)
                    if size_mask >= 1:
                        if scale and size_mask < h * w:
                            non_saliency_times = size_mask / \
                                (h * w - size_mask)
                            mae_list.append(non_saliency_times * torch.sum(torch.abs(mask_pred - mask_gt)) / size_mask)
                            self.alpha = non_saliency_times
                            is_alpha = True
                        else:
                            mae_list.append(
                                torch.sum(torch.abs(mask_pred - mask_gt)) / size_mask)
                            self.alpha = 0

            if len(mae_list) != 0:
                if is_alpha:
                    mae = sum(mae_list) / (len(mae_list) + self.alpha)
                else:
                    mae = sum(mae_list) / (len(mae_list))
                

            else:
                mae = torch.abs(pred - gt).mean()

        else:
            mae = torch.abs(pred - gt).mean()
        # breakpoint()
        return mae

    def eval_F_mean(self, pred, gt, split=None):
        # size is not considered in this metric!

        beta2 = 0.3

        def _eval_pr(pred, gt, num=100):

            prec, recall = torch.zeros(
                num, dtype=float), torch.zeros(num, dtype=float)
            # thlist = torch.linspace(0, 1 - 1e-10, num).to(self.device)
            thlist = torch.linspace(0, 1, num)

            for i in range(num):
                y_temp = (pred >= thlist[i]).float()
                tp = (y_temp * gt).sum()
                prec[i], recall[i] = tp / \
                    (y_temp.sum() + 1), tp / (gt.sum() + 1)

            return prec, recall

        pred = (pred - torch.min(pred) + 1e-20) / \
            (torch.max(pred) - torch.min(pred) + 1e-20)

        if split is not None:
            if torch.max(split) == 0:
                max_step_list = 0
            else:
                max_step_list = math.floor(
                    math.log(torch.max(split), 2) + 1) + 1
            is_alpha = False
            step_list = np.unique(split)
            f_mean_list = list()
            f_max_list = list()
            non_saliency_times = 1
            for step in range(max_step_list):
                '''
                mask_pred, mask_gt, mask = mask_split(pred, gt, split, step)
                f_prec, f_recall = _eval_pr(mask_pred, mask_gt)
                f_score = (1 + beta2) * f_prec * f_recall / (beta2 * f_prec + f_recall)
                f_score[f_score != f_score] = 0
                '''
                if step != 0:
                    mask_pred, mask_gt, mask = mask_split(
                        pred, gt, split, step)
                    if torch.sum(mask.int()) == 0:
                        continue
                    f_prec, f_recall = _eval_pr(mask_pred, mask_gt)
                    si_f_score = (1 + beta2) * f_prec * \
                        f_recall / (beta2 * f_prec + f_recall)
                    si_f_score[si_f_score != si_f_score] = 0
                    f_mean_list.append(si_f_score.mean())
                    f_max_list.append(si_f_score.max())

            if len(f_mean_list) != 0:
                return sum(f_mean_list) / len(f_mean_list), sum(f_max_list) / len(f_max_list)
            else:
                prec, recall = _eval_pr(pred, gt)
                f_score = (1 + beta2) * prec * recall / (beta2 * prec + recall)
                f_score[f_score != f_score] = 0
                return f_score.mean(), f_score.max()
        else:
            prec, recall = _eval_pr(pred, gt)
            f_score = (1 + beta2) * prec * recall / (beta2 * prec + recall)
            f_score[f_score != f_score] = 0

            return f_score.mean(), f_score.max()

    def AUROC(self, y_true, y_pred, multi_type='ova', acc=True):
        """
        Compute Area Under the Receiver Operating Characteristic Curve (AUROC).
        Note:
            This function can be only used with binary, multiclass AUC (either 'ova' or 'ovo').

        """
        # y_true = y_true.flatten()
        if not isinstance(y_true, np.ndarray):
            # warnings.warn("The type of y_ture must be np.ndarray")
            y_true = np.asarray(y_true)

        if not isinstance(y_pred, np.ndarray):
            # warnings.warn("The type of y_pred must be np.ndarray")
            y_pred = np.asarray(y_pred)

        if len(np.unique(y_true)) == 2:
            assert len(y_pred) == len(
                y_true), 'prediction and ground-truth must be the same length!'
            return roc_auc_score(y_true=y_true, y_score=y_pred)
        else:
            raise ValueError('AUROC must have at least two classes!')
    # only caculates TPR and FPR in a mask
    # auc should be computed with batch of images, where TPR_list and FPR_list are given

    def eval_auc(self, pred, gt, split=None):

        def _eval_roc(pred, gt, num=100):
            # num: threshold！
            TPR, FPR = torch.zeros(num), torch.zeros(num)
            thlist = torch.linspace(0, 1 - 1e-20, num)

            for i in range(num):
                y_temp = (pred >= thlist[i]).float()
                tp = (y_temp * gt).sum()
                fp = (y_temp * (1 - gt)).sum()
                tn = (((1 - y_temp) * (1 - gt))).sum()
                fn = ((1 - y_temp) * gt).sum()

                TPR[i] = tp / (tp + fn + 1)
                FPR[i] = fp / (fp + tn + 1)

            return TPR, FPR

        if split is not None:
            if torch.max(split) == 0:
                max_step_list = 0
            else:
                max_step_list = math.floor(
                    math.log(torch.max(split), 2) + 1) + 1
            step_list = np.unique(split)
            auc_list = list()
            # breakpoint()
            for step in range(max_step_list):
                if step != 0:
                    mask_pred, mask_gt, mask = mask_split(
                        pred, gt, split, step)
                    if torch.sum(mask.int()) == 0:
                        continue
                    TPR, FPR = _eval_roc(mask_pred, mask_gt)
                    sorted_idxes = torch.argsort(FPR)
                    avg_tpr = TPR[sorted_idxes]
                    avg_fpr = FPR[sorted_idxes]
                    auc = torch.trapz(avg_tpr, avg_fpr)
                    auc_list.append(auc)
            if len(auc_list) != 0:
                return sum(auc_list) / len(auc_list)
            else:
                TPR, FPR = _eval_roc(pred, gt)
                sorted_idxes = torch.argsort(FPR)
                avg_tpr = TPR[sorted_idxes]
                avg_fpr = FPR[sorted_idxes]
                auc = torch.trapz(avg_tpr, avg_fpr)
                return auc

        else:
            TPR, FPR = _eval_roc(pred, gt)
            sorted_idxes = torch.argsort(FPR)
            avg_tpr = TPR[sorted_idxes]
            avg_fpr = FPR[sorted_idxes]
            auc = torch.trapz(avg_tpr, avg_fpr)
            # gt[gt >= 0.5] = 1
            # gt[gt < 0.5] = 0
            # auc = self.AUROC(gt.flatten(), pred.flatten())

            return auc

    def eval_E_mean(self, pred, gt, split=None):

        def _eval_e(y_pred, y, mask=None, num=100):
            score = torch.zeros(num)
            thlist = torch.linspace(0, 1, num)
            if mask is not None:
                size = torch.sum(mask)
            for i in range(num):
                y_pred_th = (y_pred >= thlist[i]).float()
                if mask is not None:
                    fm = (y_pred_th - y_pred_th.mean()) * mask
                    gt = (y - y.mean()) * mask
                else:
                    fm = (y_pred_th - y_pred_th.mean())
                    gt = (y - y.mean())
                align_matrix = 2 * gt * fm / (gt * gt + fm * fm + 1e-20)
                # breakpoint()

                if mask is not None:
                    enhanced = mask * ((align_matrix + 1) *
                                       (align_matrix + 1)) / 4
                    score[i] = torch.sum(enhanced) / (size + 1e-20)
                else:
                    enhanced = ((align_matrix + 1) * (align_matrix + 1)) / 4
                    score[i] = torch.sum(enhanced) / (y.numel() - 1 + 1e-20)
            return score

        if split is not None:
            step_list = np.unique(split)
            E_list = list()
            for step in step_list:
                mask_pred, mask_gt, mask = mask_split(pred, gt, split, step)
                E_list.append(_eval_e(mask_pred, mask_gt, mask))
            Em = sum(E_list) / len(step_list)

        else:
            Em = _eval_e(pred, gt)

        return Em.mean()

    def eval_S(self, pred, gt, split=None):

        pred = (pred - torch.min(pred) + 1e-20) / \
            (torch.max(pred) - torch.min(pred) + 1e-20)
        alpha = 1

        def _object(pred, gt):
            temp = pred[gt == 1]
            x = temp.mean()
            sigma_x = temp.std()
            score = 2.0 * x / (x * x + 1.0 + sigma_x + 1e-20)
            return score

        def _centroid(gt, size=None):
            # print(gt.shape)
            # print(gt.size())
            # rows, cols = gt.size()[-2:]
            rows, cols = gt.size()
            gt = gt.view(rows, cols)
            if gt.sum() == 0:

                X = torch.eye(1) * round(cols / 2)
                Y = torch.eye(1) * round(rows / 2)

            else:
                total = gt.sum()

                i = torch.from_numpy(np.arange(0, cols)).float()
                j = torch.from_numpy(np.arange(0, rows)).float()

                X = torch.round((gt.sum(dim=0) * i).sum() / total + 1e-20)
                Y = torch.round((gt.sum(dim=1) * j).sum() / total + 1e-20)
            return X.long(), Y.long()

        def _divideGT(gt, X, Y, size=None):
            h, w = gt.size()[-2:]
            if size is None:
                area = h * w
            else:
                area = size
            gt = gt.view(h, w)
            LT = gt[:Y, :X]
            RT = gt[:Y, X:w]
            LB = gt[Y:h, :X]
            RB = gt[Y:h, X:w]
            X = X.float()
            Y = Y.float()
            w1 = X * Y / area
            w2 = (w - X) * Y / area
            w3 = X * (h - Y) / area
            w4 = 1 - w1 - w2 - w3
            return LT, RT, LB, RB, w1, w2, w3, w4

        def _dividePrediction(pred, X, Y):
            h, w = pred.size()[-2:]
            pred = pred.view(h, w)
            LT = pred[:Y, :X]
            RT = pred[:Y, X:w]
            LB = pred[Y:h, :X]
            RB = pred[Y:h, X:w]
            return LT, RT, LB, RB

        def _ssim(pred, gt, size=None):
            gt = gt.float()
            h, w = pred.size()[-2:]
            if size is None:
                N = h * w
            else:
                N = size
            x = pred.mean()
            y = gt.mean()
            sigma_x2 = ((pred - x) * (pred - x)).sum() / (N - 1 + 1e-20)
            sigma_y2 = ((gt - y) * (gt - y)).sum() / (N - 1 + 1e-20)
            sigma_xy = ((pred - x) * (gt - y)).sum() / (N - 1 + 1e-20)

            aplha = 4 * x * y * sigma_xy
            beta = (x * x + y * y) * (sigma_x2 + sigma_y2)

            if aplha != 0:
                Q = aplha / (beta + 1e-20)
            elif aplha == 0 and beta == 0:
                Q = 1.0
            else:
                Q = 0
            return Q

        def _S_object(pred, gt):
            fg = torch.where(gt == 0, torch.zeros_like(pred), pred)
            bg = torch.where(gt == 1, torch.zeros_like(pred), 1 - pred)
            o_fg = _object(fg, gt)
            o_bg = _object(bg, 1 - gt)
            u = gt.mean()
            Q = u * o_fg + (1 - u) * o_bg
            return Q

        def _S_region(pred, gt, size=None):
            X, Y = _centroid(gt)
            gt1, gt2, gt3, gt4, w1, w2, w3, w4 = _divideGT(gt, X, Y, size)
            p1, p2, p3, p4 = _dividePrediction(pred, X, Y)
            Q1 = _ssim(p1, gt1, size)
            Q2 = _ssim(p2, gt2, size)
            Q3 = _ssim(p3, gt3, size)
            Q4 = _ssim(p4, gt4, size)
            Q = w1 * Q1 + w2 * Q2 + w3 * Q3 + w4 * Q4
            return Q

        if split is not None:
            step_list = np.unique(split)
            S_list = list()
            for step in step_list:
                pred, gt, mask = mask_split(pred, gt, split, step)
                size = torch.sum(mask)
                y = gt.mean()
                if y == 0:
                    x = pred.mean()
                    Q = 1.0 - x
                elif y == 1:
                    x = pred.mean()
                    Q = x
                else:
                    gt[gt >= 0.5] = 1
                    gt[gt < 0.5] = 0
                    # Q = alpha * _S_object(pred, gt) + (1 - alpha) * _S_region(pred, gt)
                    Q = alpha * _S_object(pred, gt)
                    if Q.item() < 0:
                        Q = torch.FloatTensor([0.0])
                S_list.append(Q.item())
            Sm = sum(S_list) / len(step_list)

        else:
            y = gt.mean()
            if y == 0:
                x = pred.mean()
                Sm = 1.0 - x
            elif y == 1:
                x = pred.mean()
                Sm = x
            else:
                gt[gt >= 0.5] = 1
                gt[gt < 0.5] = 0
                Sm = alpha * _S_object(pred, gt) + \
                    (1 - alpha) * _S_region(pred, gt)
                if Sm.item() < 0:
                    Sm = torch.FloatTensor([0.0])

        return Sm
