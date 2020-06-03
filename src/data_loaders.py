import os
import glob
import math

import hydra
import cv2
import numpy as np
from shapely.geometry import Polygon
from torch.utils.data import Dataset, DataLoader
import imgaug.augmenters as iaa
import pyclipper

import db_transforms


def load_metadata(img_dir, gt_dir):
    img_fps = glob.glob(os.path.join(img_dir, "*"))
    gt_fps = []
    for img_fp in img_fps:
        img_id = img_fp.split("/")[-1].replace("img", "").split(".")[0]
        gt_fn = "gt_img{}.txt".format(img_id)
        gt_fp = os.path.join(gt_dir, gt_fn)
        assert os.path.exists(img_fp)
        gt_fps.append(gt_fp)
    assert len(img_fps) == len(gt_fps)

    return img_fps, gt_fps


class TotalTextDatasetIter(Dataset):
    """
    Data iteration for TotalText dataset
    """

    def __init__(self,
                 image_paths,
                 gt_paths,
                 image_size=600,
                 dataset='totaltext',
                 min_text_size=8,
                 shrink_ratio=0.4,
                 thresh_min=0.3,
                 thresh_max=0.7,
                 augment=None,
                 mean=[103.939, 116.779, 123.68],
                 debug=False):

        self.image_paths = image_paths
        self.gt_paths = gt_paths

        self.image_size = image_size
        self.min_text_size = min_text_size
        self.shrink_ratio = shrink_ratio
        self.thresh_min = thresh_min
        self.thresh_max = thresh_max
        self.augment = augment
        self.mean = mean

        self.all_anns = self.load_all_anns(gt_paths, dataset)
        assert len(self.image_paths) == len(self.all_anns)

        self.debug = debug

    def _get_default_augment(self):
        augment_seq = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Affine(rotate=(-10, 10)),
            iaa.Resize((0.5, 3.0))
        ])
        return augment_seq

    def load_all_anns(self, gt_paths, dataset='totaltext'):
        res = []
        for gt in gt_paths:
            lines = []
            reader = open(gt, 'r').readlines()
            for line in reader:
                item = {}
                parts = line.strip().split(',')
                label = parts[-1]
                line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in parts]
                num_points = math.floor((len(line) - 1) / 2) * 2
                poly = np.array(list(map(float, line[:num_points]))).reshape(
                    (-1, 2)).tolist()
                if len(poly) < 3:
                    continue
                item['poly'] = poly
                item['text'] = label
                lines.append(item)
            res.append(lines)
        return res

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):

        image_path = self.image_paths[index]
        anns = self.all_anns[index]

        if self.debug:
            print(image_path)
            print(len(anns))

        img = cv2.imread(image_path)[:, :, ::-1]
        if self.augment is not None:
            augment_seq = self.augment.to_deterministic()
            img, anns = db_transforms.transform(augment_seq, img, anns)
            img, anns = db_transforms.crop(img, anns)

        img, anns = db_transforms.resize(self.image_size, img, anns)

        anns = [ann for ann in anns if Polygon(ann['poly']).buffer(0).is_valid]
        gt = np.zeros((self.image_size, self.image_size),
                      dtype=np.float32)  # batch_gts
        mask = np.ones((self.image_size, self.image_size), dtype=np.float32)
        thresh_map = np.zeros((self.image_size, self.image_size),
                              dtype=np.float32)  # batch_thresh_maps
        # batch_thresh_masks
        thresh_mask = np.zeros((self.image_size, self.image_size),
                               dtype=np.float32)

        for ann in anns:
            poly = np.array(ann['poly'])
            height = max(poly[:, 1]) - min(poly[:, 1])
            width = max(poly[:, 0]) - min(poly[:, 0])
            polygon = Polygon(poly)

            # generate gt and mask
            if polygon.area < 1 or min(
                    height, width) < self.min_text_size or ann['text'] == '#':
                cv2.fillPoly(mask, poly.astype(np.int32)[np.newaxis, :, :], 0)
                continue
            else:
                # 6th equation
                distance = polygon.area * \
                    (1 - np.power(self.shrink_ratio, 2)) / polygon.length
                subject = [tuple(_l) for _l in ann['poly']]
                padding = pyclipper.PyclipperOffset()
                padding.AddPath(subject, pyclipper.JT_ROUND,
                                pyclipper.ET_CLOSEDPOLYGON)
                shrinked = padding.Execute(-distance)

                if len(shrinked) == 0:
                    cv2.fillPoly(mask,
                                 poly.astype(np.int32)[np.newaxis, :, :], 0)
                    continue
                else:
                    shrinked = np.array(shrinked[0]).reshape(-1, 2)
                    if shrinked.shape[0] > 2 and Polygon(shrinked).buffer(0).is_valid:  # noqa
                        cv2.fillPoly(gt, [shrinked.astype(np.int32)], 1)
                    else:
                        cv2.fillPoly(mask,
                                     poly.astype(np.int32)[np.newaxis, :, :],
                                     0)
                        continue

            # generate thresh map and thresh mask
            db_transforms.draw_thresh_map(ann['poly'],
                                          thresh_map,
                                          thresh_mask,
                                          shrink_ratio=self.shrink_ratio)

        thresh_map = thresh_map * \
            (self.thresh_max - self.thresh_min) + self.thresh_min

        img = img.astype(np.float32)
        img[..., 0] -= self.mean[0]
        img[..., 1] -= self.mean[1]
        img[..., 2] -= self.mean[2]

        img = np.transpose(img, (2, 0, 1))

        # resized_image, prob_map, supervision_mask, threshold_map, text_area_map
        return image_path, img, gt, mask, thresh_map, thresh_mask


@hydra.main(config_path="../config.yaml", strict=False)
def run(cfg):
    image_paths, gt_paths = load_metadata(
        cfg.data.totaltext.train_dir,
        cfg.data.totaltext.train_gt_dir
    )
    totaltext_train_iter = TotalTextDatasetIter(image_paths,
                                                gt_paths,
                                                debug=False)
    totaltext_train_loader = DataLoader(
        dataset=totaltext_train_iter,
        batch_size=2,
        shuffle=True,
        num_workers=1
    )
    print(len(next(iter(totaltext_train_loader))))


if __name__ == '__main__':
    run()
