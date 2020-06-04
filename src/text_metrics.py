# Adapted from score written by wkentaro
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py
import numpy as np


class running_score:
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)

        if np.sum((label_pred[mask] < 0)) > 0:
            print(label_pred[label_pred < 0])
        hist = np.bincount(n_class * label_true[mask].astype(int) +
                           label_pred[mask],
                           minlength=n_class**2).reshape(n_class, n_class)

        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            try:
                self.confusion_matrix += self._fast_hist(
                    lt.flatten(), lp.flatten(), self.n_classes)
            except Exception as e:
                print(e)
                pass

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / (hist.sum() + 0.0001)
        acc_cls = np.diag(hist) / (hist.sum(axis=1) + 0.0001)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) -
                              np.diag(hist) + 0.0001)
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / (hist.sum() + 0.0001)
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return {
            'Overall Acc': acc,
            'Mean Acc': acc_cls,
            'FreqW Acc': fwavacc,
            'Mean IoU': mean_iu,
        }, cls_iu

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


def cal_text_score(texts,
                   gt_texts,
                   training_masks,
                   running_metric_text,
                   thred=0.5):
    """
    :param texts: preb_prob_map
    :param gt_texts: gt_prob_map
    :param training_masks: supervision map
    """
    training_masks = training_masks.data.cpu().numpy()
    pred_text = texts.data.cpu().numpy() * training_masks
    pred_text[pred_text <= thred] = 0
    pred_text[pred_text > thred] = 1
    pred_text = pred_text.astype(np.int32)
    gt_text = gt_texts.data.cpu().numpy() * training_masks
    gt_text = gt_text.astype(np.int32)
    running_metric_text.update(gt_text, pred_text)
    score_text, _ = running_metric_text.get_scores()
    return score_text
