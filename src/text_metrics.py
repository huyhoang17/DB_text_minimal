# Adapted from score written by wkentaro
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py
import numpy as np

from iou import DetectionIoUEvaluator
from utils import to_list_tuples_coords


class RunningScore:
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
                   thresh=0.5):
    """
    :param texts: preb_prob_map
    :param gt_texts: gt_prob_map
    :param training_masks: supervision map
    """
    training_masks = training_masks.data.cpu().numpy()
    pred_text = texts.data.cpu().numpy() * training_masks
    pred_text[pred_text <= thresh] = 0
    pred_text[pred_text > thresh] = 1
    pred_text = pred_text.astype(np.int32)
    gt_text = gt_texts.data.cpu().numpy() * training_masks
    gt_text = gt_text.astype(np.int32)
    running_metric_text.update(gt_text, pred_text)
    score_text, _ = running_metric_text.get_scores()
    return score_text


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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
        return self


class QuadMetric:
    def __init__(self):
        self.evaluator = DetectionIoUEvaluator()

    def measure(self, batch, output, is_output_polygon=False, box_thresh=0.6):
        '''
        batch: (image, polygons, ignore_tags
        batch: a dict produced by dataloaders.
            image: tensor of shape (N, C, H, W).
            polygons: tensor of shape (N, K, 4, 2), the polygons of objective regions.
            ignore_tags: tensor of shape (N, K), indicates whether a region is ignorable or not.
            shape: the original shape of images.
            filename: the original filenames of images.
        output: (polygons, ...)
        '''
        results = []
        # gt_polyons_batch = batch['text_polys']
        # ignore_tags_batch = batch['ignore_tags']

        # for each image
        # for each polygon in image
        # coordinates

        pred_polygons_batch = np.array(output[0])
        pred_scores_batch = np.array(output[1])
        # print(pred_polygons_batch.shape, pred_scores_batch.shape)
        # print(np.array(pred_polygons_batch[0]).shape)
        # print(np.array(pred_polygons_batch[0][0]).shape)
        # print(np.array(pred_scores_batch[0][0]))

        # gts = [
        #     [
        #         {
        #             'points': [
        #                 (0, 0), (1, 0), (1, 1), (0, 1),
        #                 # (-1, 1),
        #             ],
        #             'text': 1234,
        #             'ignore': False,
        #         },
        #         {
        #             'points': [(2, 2), (3, 2), (3, 3), (2, 3)],
        #             'text': 5678,
        #             'ignore': True,
        #         }
        #     ]
        # ]
        gt_polygons_batch = to_list_tuples_coords(batch['anns'])
        ignore_tags_batch = [i[0].tolist() for i in batch['ignore_tags']]
        gt = []
        for gt_polygon, ignore_tag in zip(gt_polygons_batch,
                                          ignore_tags_batch):
            gt.append({'points': gt_polygon, 'ignore': ignore_tag})

        pred = []  # for 1 image
        for pred_polygon, pred_score in zip(pred_polygons_batch[0],
                                            pred_scores_batch[0]):
            pred.append({'points': pred_polygon, 'ignore': False})
        results.append(self.evaluator.evaluate_image(gt, pred))

        # 4 points only!!!
        # for polygons, pred_polygons, pred_scores, ignore_tags in zip(
        #         gt_polyons_batch, pred_polygons_batch, pred_scores_batch,
        #         ignore_tags_batch):
        #     gt = [
        #         dict(
        #             points=np.int64(polygons[i]),
        #             # ignore=ignore_tags[i],
        #             ignore=True if ignore_tags[i] == '#' else False
        #         )
        #         for i in range(len(polygons))
        #     ]
        #     if is_output_polygon:
        #         pred = [
        #             dict(points=pred_polygons[i])
        #             for i in range(len(pred_polygons))
        #         ]
        #     else:
        #         pred = []
        #         # print(pred_polygons.shape)
        #         for i in range(pred_polygons.shape[0]):
        #             if pred_scores[i] >= box_thresh:
        #                 # print(pred_polygons[i,:,:].tolist())
        #                 pred.append(
        #                     dict(points=pred_polygons[i, :, :].astype(np.int)))
        #         # pred = [dict(points=pred_polygons[i,:,:].tolist()) if pred_scores[i] >= box_thresh for i in range(pred_polygons.shape[0])]
        #     results.append(self.evaluator.evaluate_image(gt, pred))
        return results

    def validate_measure(self,
                         batch,
                         output,
                         is_output_polygon=False,
                         box_thresh=0.6):
        return self.measure(batch, output, is_output_polygon, box_thresh)

    # def evaluate_measure(self, batch, output):
    #     return self.measure(batch, output), np.linspace(
    #         0, batch['image'].shape[0]).tolist()

    def gather_measure(self, raw_metrics):
        raw_metrics = [
            image_metrics for batch_metrics in raw_metrics
            for image_metrics in batch_metrics
        ]

        result = self.evaluator.combine_results(raw_metrics)

        precision = AverageMeter()
        recall = AverageMeter()
        fmeasure = AverageMeter()

        precision.update(result['precision'], n=len(raw_metrics))
        recall.update(result['recall'], n=len(raw_metrics))
        fmeasure_score = 2 * precision.val * recall.val / (precision.val +
                                                           recall.val + 1e-8)
        fmeasure.update(fmeasure_score)

        return {'precision': precision, 'recall': recall, 'fmeasure': fmeasure}
