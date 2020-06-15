import torch
import torch.nn as nn


def step_function(x, y, k=50):
    x = torch.Tensor(x)
    y = torch.Tensor(y)
    return torch.reciprocal(1 + torch.exp(k * (x - y)))


class OHEMBalanceCrossEntropyLoss(nn.Module):
    def __init__(self, negative_ratio=3, eps=1e-6, reduction='mean'):
        super().__init__()
        self.negative_ratio = negative_ratio
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred, gt, mask):
        """
        :params: pred_prob, gt_prob, supervision_map
        """
        positive = (gt * mask)
        negative = ((1 - gt) * mask)

        no_positive = int(positive.sum())
        no_negative_expect = int(no_positive * self.negative_ratio)
        no_negative_current = int(negative.sum())
        no_negative = min(no_negative_expect, no_negative_current)

        loss = nn.functional.binary_cross_entropy(pred,
                                                  gt,
                                                  reduction=self.reduction)
        positive_loss = loss * positive
        negative_loss = loss * negative

        negative_loss, _ = torch.topk(negative_loss.view(-1), no_negative)

        balance_loss = (positive_loss.sum() + negative_loss.sum()) / (
            no_positive + no_negative + self.eps)
        return balance_loss


class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred, gt, mask, weights=None):
        """
        :params: appro binary map, gt_prob, supervision map
        """
        #         if pred.dim() == 4:
        #             pred = pred[:, 0, :, :]
        #             gt = gt[:, 0, :, :]
        #         assert pred.shape == gt.shape
        #         assert pred.shape == mask.shape

        #         if weights is not None:
        #             assert mask.shape == weights.shape
        #             mask = weights * mask

        intersection = (pred * gt * mask).sum()
        union = (pred * mask).sum() + (gt * mask).sum() + self.eps
        loss = 1 - 2.0 * intersection / union
        assert loss <= 1
        return loss


class L1Loss(nn.Module):
    def __init__(self, eps=1e-6, reduction='mean'):
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred, gt, mask):
        if mask is not None:
            loss = (torch.abs(pred - gt) * mask).sum() / \
                (mask.sum() + self.eps)
        else:
            l1_loss_fn = torch.nn.L1Loss(reduction=self.reduction)
            loss = l1_loss_fn(pred, gt)
        return loss


class DBLoss(nn.Module):
    def __init__(self,
                 alpha=1.0,
                 beta=10.0,
                 reduction='mean',
                 negative_ratio=3,
                 eps=1e-6):
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction
        self.negative_ratio = negative_ratio
        self.eps = eps

        self.ohem_loss = OHEMBalanceCrossEntropyLoss(self.negative_ratio,
                                                     self.eps, self.reduction)
        self.dice_loss = DiceLoss(self.eps)
        self.l1_loss = L1Loss(self.eps, self.reduction)

    def forward(self, preds, gts):
        """
        :params: preds (train mode): prob map, thresh map, binary map
        :params: gts (eval mode): prob map, thresh map
        """

        # predicts
        # prob_map, threshold_map, binary_map
        assert preds.dim() == 4
        assert gts.dim() == 4

        prob_pred = preds[:, 0, :, :]
        threshold_map = preds[:, 1, :, :]
        if preds.size(1) == 3:
            appro_binary_map = preds[:, 2, :, :]  # dim = 3

        # ground truths
        # prob_map, supervision_mask, threshold_map, text_area_map
        prob_gt_map = gts[0, :, :, :]  # 0/1
        supervision_mask = gts[1, :, :, :]  # 0/1
        threshold_gt_map = gts[2, :, :, :]  # 0.3 -> 0.7
        text_area_gt_map = gts[3, :, :, :]  # 0/1

        # losses
        prob_loss = self.ohem_loss(prob_pred, prob_gt_map, supervision_mask)
        threshold_loss = self.l1_loss(threshold_map, threshold_gt_map,
                                      text_area_gt_map)
        prob_threshold_loss = prob_loss + self.beta * threshold_loss
        if preds.size(1) == 3:
            binary_loss = self.dice_loss(appro_binary_map, prob_gt_map,
                                         supervision_mask)
            total_loss = self.alpha * binary_loss + prob_threshold_loss
            return prob_loss, threshold_loss, binary_loss, prob_threshold_loss, total_loss  # noqa
        else:
            return prob_threshold_loss
