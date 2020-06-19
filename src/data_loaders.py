import os
import glob
import math

import hydra
import cv2
import numpy as np
from shapely.geometry import Polygon
import torch
from torch.utils.data import Dataset, DataLoader
import imgaug.augmenters as iaa
import pyclipper

import db_transforms
from utils import dict_to_device, minmax_scaler_img


class BaseDatasetIter(Dataset):
    def __init__(self,
                 train_dir,
                 train_gt_dir,
                 ignore_tags,
                 is_training=True,
                 image_size=640,
                 min_text_size=8,
                 shrink_ratio=0.4,
                 thresh_min=0.3,
                 thresh_max=0.7,
                 augment=None,
                 mean=[103.939, 116.779, 123.68],
                 debug=False):

        self.train_dir = train_dir
        self.train_gt_dir = train_gt_dir
        self.ignore_tags = ignore_tags

        self.is_training = is_training
        self.image_size = image_size
        self.min_text_size = min_text_size
        self.shrink_ratio = shrink_ratio
        self.thresh_min = thresh_min
        self.thresh_max = thresh_max
        self.augment = augment
        if self.augment is None:
            self.augment = self._get_default_augment()

        self.mean = mean
        self.debug = debug

        # load metadata
        self.image_paths, self.gt_paths = self.load_metadata(
            train_dir, train_gt_dir)

        # load annotation for totaltext dataset
        self.all_anns = self.load_all_anns(self.gt_paths)
        assert len(self.image_paths) == len(self.all_anns)

    def _get_default_augment(self):
        augment_seq = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Affine(rotate=(-10, 10)),
            iaa.Resize((0.5, 3.0))
        ])
        return augment_seq

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):

        image_path = self.image_paths[index]
        anns = self.all_anns[index]

        if self.debug:
            print(image_path)
            print(len(anns))

        img = cv2.imread(image_path)[:, :, ::-1]
        if self.is_training and self.augment is not None:
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

        if self.debug:
            print(type(anns), len(anns))

        ignore_tags = []
        for ann in anns:
            # i.e shape = (4, 2) / (6, 2) / ...
            poly = np.array(ann['poly'])
            height = max(poly[:, 1]) - min(poly[:, 1])
            width = max(poly[:, 0]) - min(poly[:, 0])
            polygon = Polygon(poly)

            # generate gt and mask
            if polygon.area < 1 or \
                    min(height, width) < self.min_text_size or \
                    ann['text'] in self.ignore_tags:
                ignore_tags.append(True)
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
                    ignore_tags.append(True)
                    cv2.fillPoly(mask,
                                 poly.astype(np.int32)[np.newaxis, :, :], 0)
                    continue
                else:
                    shrinked = np.array(shrinked[0]).reshape(-1, 2)
                    if shrinked.shape[0] > 2 and \
                            Polygon(shrinked).buffer(0).is_valid:
                        ignore_tags.append(False)
                        cv2.fillPoly(gt, [shrinked.astype(np.int32)], 1)
                    else:
                        ignore_tags.append(True)
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

        data_return = {
            "image_path": image_path,
            "img": img,
            "prob_map": gt,
            "supervision_mask": mask,
            "thresh_map": thresh_map,
            "text_area_map": thresh_mask,
        }
        # for batch_size = 1
        if not self.is_training:
            data_return["anns"] = [ann['poly'] for ann in anns]
            data_return["ignore_tags"] = ignore_tags

        # return image_path, img, gt, mask, thresh_map, thresh_mask
        return data_return


class TotalTextDatasetIter(BaseDatasetIter):
    """
    Data iteration for TotalText dataset
    """
    def __init__(self, train_dir, train_gt_dir, ignore_tags, **kwargs):
        super().__init__(train_dir, train_gt_dir, ignore_tags, **kwargs)

    def load_metadata(self, img_dir, gt_dir):
        img_fps = sorted(glob.glob(os.path.join(img_dir, "*")))
        gt_fps = []
        for img_fp in img_fps:
            img_id = img_fp.split("/")[-1].replace("img", "").split(".")[0]
            gt_fn = "gt_img{}.txt".format(img_id)
            gt_fp = os.path.join(gt_dir, gt_fn)
            assert os.path.exists(img_fp)
            gt_fps.append(gt_fp)
        assert len(img_fps) == len(gt_fps)

        return img_fps, gt_fps

    def load_all_anns(self, gt_paths):
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


class CTW1500DatasetIter(BaseDatasetIter):
    def __init__(self, train_dir, train_gt_dir, ignore_tags, **kwargs):
        super().__init__(train_dir, train_gt_dir, ignore_tags, **kwargs)

    def load_metadata(self, img_dir, gt_dir):
        img_fps = sorted(glob.glob(os.path.join(img_dir, "*")))
        gt_fps = []
        for img_fp in img_fps:
            img_id = img_fp.split("/")[-1][:-4]
            gt_fn = "{}.txt".format(img_id)
            gt_fp = os.path.join(gt_dir, gt_fn)
            assert os.path.exists(img_fp)
            gt_fps.append(gt_fp)
        assert len(img_fps) == len(gt_fps)

        return img_fps, gt_fps

    def load_all_anns(self, gt_fps):
        """
        Reference: https://github.com/whai362/PSENet/blob/master/dataset/ctw1500_loader.py
        """
        res = []
        for gt_fp in gt_fps:
            lines = []
            with open(gt_fp, 'r') as f:
                for line in f:
                    item = {}
                    gt = line.strip().strip('\ufeff').strip('\xef\xbb\xbf')
                    gt = list(map(int, gt.split(',')))

                    x1 = np.int(gt[0])
                    y1 = np.int(gt[1])
                    bbox = [np.int(gt[i]) for i in range(4, 32)]
                    bbox = np.asarray(bbox) + ([x1 * 1, y1 * 1] * 14)
                    bbox = bbox.reshape(-1, 2).tolist()
                    item['poly'] = bbox
                    item['text'] = 'True'
                    lines.append(item)
            res.append(lines)
        return res


class ICDAR2015DatasetIter(BaseDatasetIter):
    def __init__(self, train_dir, train_gt_dir, ignore_tags, **kwargs):
        super().__init__(train_dir, train_gt_dir, ignore_tags, **kwargs)

    def load_metadata(self, img_dir, gt_dir):
        img_fps = glob.glob(os.path.join(img_dir, "*"))
        gt_fps = []
        for img_fp in img_fps:
            img_id = img_fp.split("/")[-1].split(".")[0]
            gt_fn = "gt_{}.txt".format(img_id)
            gt_fp = os.path.join(gt_dir, gt_fn)
            assert os.path.exists(img_fp)
            gt_fps.append(gt_fp)
        assert len(img_fps) == len(gt_fps)

        return img_fps, gt_fps

    def load_all_anns(self, gt_fps):
        res = []
        for gt_fp in gt_fps:
            lines = []
            with open(gt_fp, 'r') as f:
                for line in f:
                    item = {}
                    gt = line.strip().strip('\ufeff').strip(
                        '\xef\xbb\xbf').split(",")
                    label = ",".join(gt[8:])
                    poly = list(map(int, gt[:8]))
                    poly = np.asarray(poly).reshape(-1, 2).tolist()
                    item['poly'] = poly
                    item['text'] = label
                    lines.append(item)
                import pdb
                pdb.set_trace()
            res.append(lines)
        return res


class MSRATD500DatasetIter(BaseDatasetIter):
    def __init__(self, train_dir, train_gt_dir, ignore_tags, **kwargs):
        super().__init__(train_dir, train_gt_dir, ignore_tags, **kwargs)

    def transform_four_points(self, points, center_point, theta):
        """Reference: https://stackoverflow.com/questions/622140
        """
        theta = -theta
        new_coords = []
        x_center, y_center = center_point

        for point in points:
            x, y = point
            x_new = x_center + (x - x_center) * np.cos(theta) + \
                (y - y_center) * np.sin(theta)
            y_new = y_center - (x - x_center) * np.sin(theta) + \
                (y - y_center) * np.cos(theta)
            x_new = int(x_new)
            y_new = int(y_new)
            new_coords.append((x_new, y_new))
        return new_coords

    def load_metadata(self, img_dir, gt_dir=None):
        # ignore gt_dir
        img_fps = sorted(glob.glob(os.path.join(img_dir, "*.JPG")))
        gt_fps = sorted(glob.glob(os.path.join(img_dir, "*.gt")))
        assert len(img_fps) == len(gt_fps)

        return img_fps, gt_fps

    def load_all_anns(self, gt_fps):
        res = []
        for gt_fp in gt_fps:
            lines = []
            with open(gt_fp, 'r') as f:
                for line in f:
                    item = {}
                    line = list(map(float, line.strip().split()))
                    index, dif, x_min, y_min, w, h, theta = line
                    if int(dif) == 1:  # difficult label
                        continue

                    c1 = (x_min, y_min)
                    c2 = (x_min + w, y_min)
                    c3 = (x_min + w, y_min + h)
                    c4 = (x_min, y_min + h)
                    center = (x_min + w / 2, y_min + h / 2)
                    rot_box = self.transform_four_points([c1, c2, c3, c4],
                                                         center, theta)
                    rot_box = np.array(rot_box).tolist()

                    item['poly'] = rot_box
                    item['text'] = 'True'
                    lines.append(item)
            res.append(lines)
        return res


@hydra.main(config_path="../config.yaml", strict=False)
def run(cfg):
    dataset_name = cfg.dataset.name
    ignore_tags = cfg.data[dataset_name].ignore_tags
    train_dir = cfg.data[dataset_name].train_dir
    train_gt_dir = cfg.data[dataset_name].train_gt_dir

    if dataset_name == 'totaltext':
        TextDatasetIter = TotalTextDatasetIter
    elif dataset_name == 'ctw1500':
        TextDatasetIter = CTW1500DatasetIter
    elif dataset_name == 'icdar2015':
        TextDatasetIter = ICDAR2015DatasetIter
    elif dataset_name == 'msra_td500':
        TextDatasetIter = MSRATD500DatasetIter
    else:
        raise NotImplementedError("Pls provide valid dataset name!")
    train_iter = TextDatasetIter(train_dir,
                                 train_gt_dir,
                                 ignore_tags,
                                 is_training=True,
                                 debug=False)
    train_loader = DataLoader(dataset=train_iter,
                              batch_size=1,
                              shuffle=True,
                              num_workers=1)
    samples = next(iter(train_loader))
    samples = dict_to_device(samples, device='cpu')
    for k, v in samples.items():
        if isinstance(v, torch.Tensor):
            print(samples[k].device)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(minmax_scaler_img(samples['img'][0].numpy().transpose(1, 2, 0)))
    plt.imshow(samples['prob_map'][0], cmap='jet', alpha=0.35)
    plt.imshow(samples['thresh_map'][0], cmap='jet', alpha=0.5)
    # plt.imshow(samples['text_area_map'][0], cmap='jet', alpha=0.5)
    # plt.imshow(samples['supervision_mask'][0], cmap='jet', alpha=0.5)
    plt.savefig(os.path.join(cfg.meta.root_dir, 'tmp/foo.jpg'),
                bbox_inches='tight')


if __name__ == '__main__':
    run()
