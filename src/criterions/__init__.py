from .metrics import Metric
import torch
# from .eval import SalEval


class SalEval:
    def __init__(self) -> None:
        self.metric = Metric()
        self.MAX_IMG_PER_BATCH = 256
        self.num_images = 0
        self.mae = 0.0
        self.si_mae = 0.0
        self.si_mean_F = 0.0
        self.si_max_F = 0.0
        self.auc = 0.0
        self.si_auc = 0.0
        self.tpr = 0.0
        self.fpr = 0.0
        self.mean_F = 0.0
        self.max_F = 0.0
        self.Em = 0.0
        self.Sm = 0.0

    def eval_mae(self, pred, gt, split=None, scale=False):
        batch = pred.shape[0]
        mae = 0
        # print(pred.shape)
        # print(gt.shape)
        if split is None:
            for idx in range(batch):
                mae += self.metric.eval_mae(pred[idx], gt[idx], None, scale)
        else:
            for idx in range(batch):
                mae += self.metric.eval_mae(pred[idx],
                                            gt[idx], split[idx], scale)
        return mae

    def eval_auc(self, pred, gt, split=None):
        batch = pred.shape[0]
        fpr, tpr = 0.0, 0.0
        auc = 0.0
        if split is None:
            for idx in range(batch):
                auc += self.metric.eval_auc(pred[idx], gt[idx], None)
                # fpr += f
                # tpr += t
        else:
            for idx in range(batch):
                auc += self.metric.eval_auc(pred[idx], gt[idx], split[idx])
                # fpr += f
                # tpr += t
        # return fpr, tpr
        return auc

    def eval_mean_F(self, pred, gt, split=None):
        batch = pred.shape[0]
        mean_F = 0
        max_F = 0
        if split is None:
            for idx in range(batch):
                mean_f, max_f = self.metric.eval_F_mean(
                    pred[idx], gt[idx], None)
                mean_F += mean_f
                max_F += max_f
        else:
            for idx in range(batch):
                mean_f, max_f = self.metric.eval_F_mean(
                    pred[idx], gt[idx], split[idx])
                mean_F += mean_f
                max_F += max_f
        return mean_F, max_F

    def eval_Em(self, pred, gt, split=None):
        batch = pred.shape[0]
        Em = 0
        if split is None:
            for idx in range(batch):
                Em += self.metric.eval_E_mean(pred[idx], gt[idx], None)
        else:
            for idx in range(batch):
                Em += self.metric.eval_E_mean(pred[idx], gt[idx], split[idx])
        return Em

    def eval_Sm(self, pred, gt, split=None):
        batch = pred.shape[0]
        Sm = 0
        if split is None:
            for idx in range(batch):
                Sm += self.metric.eval_S(pred[idx], gt[idx], None)
        else:
            for idx in range(batch):
                Sm += self.metric.eval_S(pred[idx], gt[idx], split[idx])
        return Sm

    def add_batch(self, pred, gt, metrics, split=None, scale=False):
        # eps = 1e-20
        # pred = (pred - torch.min(pred) + eps) / (torch.max(pred) - torch.min(pred) + eps)
        batch = pred.shape[0]
        assert (pred.shape[0] < self.MAX_IMG_PER_BATCH)

        if metrics["mae"]:
            self.mae += self.eval_mae(pred, gt, None, scale)
        if metrics["si_mae"]:
            self.si_mae += self.eval_mae(pred, gt, split, scale)
        if metrics["auc"]:
            self.auc += self.eval_auc(pred, gt, None)
        if metrics["si_auc"]:
            self.si_auc += self.eval_auc(pred, gt, split)
        if metrics["f"]:
            mean_f, max_f = self.eval_mean_F(pred, gt, None)
            self.mean_F += mean_f
            self.max_F += max_f
        if metrics["si_f"]:
            si_mean_F, si_max_F = self.eval_mean_F(pred, gt, split)
            self.si_mean_F += si_mean_F
            self.si_max_F += si_max_F
        if metrics["e"]:
            self.Em += self.eval_Em(pred, gt, None)

        # self.Sm += self.eval_Sm(pred, gt, split)

        self.num_images += batch

    def get_single(self, pred, gt,  metrics, split=None):
        mae, si_mae, avg_auc, si_avg_auc, mean_F, max_F, si_mean_F, si_max_F, Em = 1, 1, 0, 0, 0, 0, 0, 0, 0
        if metrics["mae"]:
            mae = self.eval_mae(pred, gt, None)
        if metrics["si_mae"]:
            si_mae = self.eval_mae(pred, gt, split)
        if metrics["auc"]:
            avg_auc = self.eval_auc(pred, gt, None)
        if metrics["si_auc"]:
            si_avg_auc = self.eval_auc(pred, gt, split)
        if metrics["f"]:
            mean_f, max_f = self.eval_mean_F(pred, gt, None)
            mean_F = mean_f
            max_F = max_f
        if metrics["si_f"]:
            si_mean_F, si_max_F = self.eval_mean_F(pred, gt, split)
        if metrics["e"]:
            Em = self.eval_Em(pred, gt, None)

        return mae, si_mae, avg_auc, si_avg_auc, mean_F, max_F, si_mean_F, si_max_F, Em

    def get_result(self, metrics):

        if metrics["mae"]:
            mae = self.mae / self.num_images
        else:
            mae = torch.tensor(0)
        if metrics["si_mae"]:
            si_mae = self.si_mae / self.num_images
        else:
            si_mae = torch.tensor(0)
        if metrics["auc"]:
            avg_auc = (self.auc) / self.num_images
        else:
            avg_auc = torch.tensor(0)
        if metrics["si_auc"]:
            si_avg_auc = (self.si_auc) / self.num_images
        else:
            si_avg_auc = torch.tensor(0)
        if metrics["f"]:
            mean_F = (self.mean_F) / self.num_images
            max_F = (self.max_F) / self.num_images
        else:
            mean_F, max_F = torch.tensor(0), torch.tensor(0)
        if metrics["si_f"]:
            si_mean_F = (self.si_mean_F) / self.num_images
            si_max_F = (self.si_max_F) / self.num_images
        else:
            si_mean_F, si_max_F = torch.tensor(0), torch.tensor(0)
        if metrics["e"]:
            Em = self.Em / self.num_images
        else:
            Em = torch.tensor(0)

        # Sm = self.Sm / self.num_images

        return mae, si_mae, avg_auc, si_avg_auc, mean_F, max_F, si_mean_F, si_max_F, Em


def get_evaluator():
    evaluator = SalEval()
    return evaluator
