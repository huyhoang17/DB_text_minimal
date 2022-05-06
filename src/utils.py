import os
import gc
import glob
import time
import random
import imageio
import logging
from functools import wraps

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.utils as torch_utils

from postprocess import SegDetectorRepresenter

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'


def setup_determinism(seed=42):
    """
    https://github.com/pytorch/pytorch/issues/7068#issuecomment-487907668
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def setup_logger(logger_name='dbtext', log_file_path=None):
    logging._warn_preinit_stderr = 0
    logger = logging.getLogger(logger_name)
    formatter = logging.Formatter(
        '%(asctime)s %(name)s %(levelname)s: %(message)s')

    if log_file_path is not None:
        file_handle = logging.FileHandler(log_file_path)
        file_handle.setFormatter(formatter)
        logger.addHandler(file_handle)
    logger.setLevel(logging.DEBUG)

    return logger


def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f">>> Function {func.__name__}: {end - start}'s")
        return result

    return wrapper


def to_device(batch, device='cuda'):
    new_batch = []

    for ele in batch:
        if isinstance(ele, torch.Tensor):
            new_batch.append(ele.to(device))
        else:
            new_batch.append(ele)
    return new_batch


def dict_to_device(batch, device='cuda'):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
    return batch


def to_list_tuples_coords(anns):
    new_anns = []
    for ann in anns:
        points = [(x[0].tolist(), y[0].tolist()) for x, y in ann]
        new_anns.append(points)
    return new_anns


def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


def str_to_bool(value):
    if value.lower() in {'False', 'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'True', 'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


def minmax_scaler_img(img):
    img = ((img - img.min()) * (1 / (img.max() - img.min()) * 255)).astype(
        'uint8')  # noqa
    return img


def visualize_tfb(tfb_writer,
                  imgs,
                  preds,
                  global_steps,
                  thresh=0.5,
                  mode="TRAIN"):
    # origin img
    # imgs.shape = (batch_size, 3, image_size, image_size)
    imgs = torch.stack([
        torch.Tensor(
            minmax_scaler_img(img_.to('cpu').numpy().transpose((1, 2, 0))))
        for img_ in imgs
    ])
    imgs = torch.Tensor(imgs.numpy().transpose((0, 3, 1, 2)))
    imgs_grid = torch_utils.make_grid(imgs)
    imgs_grid = torch.unsqueeze(imgs_grid, 0)
    # imgs_grid.shape = (3, image_size, image_size * batch_size)
    tfb_writer.add_images(f'{mode}/origin_imgs', imgs_grid, global_steps)

    # pred_prob_map / pred_thresh_map
    pred_prob_map = preds[:, 0, :, :]
    pred_thred_map = preds[:, 1, :, :]
    pred_prob_map[pred_prob_map <= thresh] = 0
    pred_prob_map[pred_prob_map > thresh] = 1

    # make grid
    pred_prob_map = pred_prob_map.unsqueeze(1)
    pred_thred_map = pred_thred_map.unsqueeze(1)

    probs_grid = torch_utils.make_grid(pred_prob_map, padding=0)
    probs_grid = torch.unsqueeze(probs_grid, 0)
    probs_grid = probs_grid.detach().to('cpu')

    thres_grid = torch_utils.make_grid(pred_thred_map, padding=0)
    thres_grid = torch.unsqueeze(thres_grid, 0)
    thres_grid = thres_grid.detach().to('cpu')

    tfb_writer.add_images(f'{mode}/prob_imgs', probs_grid, global_steps)
    tfb_writer.add_images(f'{mode}/thres_imgs', thres_grid, global_steps)


def test_resize(img, size=640, pad=False):
    h, w, c = img.shape
    scale_w = size / w
    scale_h = size / h
    scale = min(scale_w, scale_h)
    h = int(h * scale)
    w = int(w * scale)

    new_img = None
    if pad:
        new_img = np.zeros((size, size, c), img.dtype)
        new_img[:h, :w] = cv2.resize(img, (w, h))
    else:
        new_img = cv2.resize(img, (w, h))

    return new_img


def read_img(img_fp):
    img = cv2.imread(img_fp)[:, :, ::-1]
    h_origin, w_origin, _ = img.shape
    return img, h_origin, w_origin


def test_preprocess(img,
                    mean=[103.939, 116.779, 123.68],
                    to_tensor=True,
                    pad=False):
    img = test_resize(img, size=640, pad=pad)

    img = img.astype(np.float32)
    img[..., 0] -= mean[0]
    img[..., 1] -= mean[1]
    img[..., 2] -= mean[2]
    img = np.expand_dims(img, axis=0)

    if to_tensor:
        img = torch.Tensor(img.transpose(0, 3, 1, 2))

    return img


def draw_bbox(img, result, color=(255, 0, 0), thickness=3):
    """
    :input: RGB img
    """
    if isinstance(img, str):
        img = cv2.imread(img)
    img = img.copy()
    for point in result:
        point = point.astype(int)
        cv2.polylines(img, [point], True, color, thickness)
    return img


def visualize_heatmap(args, img_fn, tmp_img, tmp_pred):
    pred_prob = tmp_pred[0]
    pred_prob[pred_prob <= args.prob_thred] = 0
    pred_prob[pred_prob > args.prob_thred] = 1

    np_img = minmax_scaler_img(tmp_img[0].to(device).numpy().transpose(
        (1, 2, 0)))
    plt.imshow(np_img)
    plt.imshow(pred_prob, cmap='jet', alpha=args.alpha)
    img_fn = f"heatmap_result_{img_fn}"
    plt.savefig(os.path.join(args.save_dir, img_fn),
                dpi=200,
                bbox_inches='tight')
    gc.collect()


def visualize_polygon(args, img_fn, origin_info, batch, preds, vis_char=False):
    img_origin, h_origin, w_origin = origin_info
    seg_obj = SegDetectorRepresenter(thresh=args.thresh,
                                     box_thresh=args.box_thresh,
                                     unclip_ratio=args.unclip_ratio)
    box_list, score_list = seg_obj(batch,
                                   preds,
                                   is_output_polygon=args.is_output_polygon)
    box_list, score_list = box_list[0], score_list[0]

    if len(box_list) > 0:
        if args.is_output_polygon:
            idx = [x.sum() > 0 for x in box_list]
            box_list = [box_list[i] for i, v in enumerate(idx) if v]
            score_list = [score_list[i] for i, v in enumerate(idx) if v]
        else:
            idx = box_list.reshape(box_list.shape[0], -1).sum(axis=1) > 0
            box_list, score_list = box_list[idx], score_list[idx]
    else:
        box_list, score_list = [], []

    tmp_img = draw_bbox(img_origin, np.array(box_list))
    tmp_pred = cv2.resize(preds[0, 0, :, :].cpu().numpy(),
                          (w_origin, h_origin))

    if not args.is_output_polygon and vis_char:

        char_img_fps = glob.glob(os.path.join("./tmp/reconized", "*"))
        for char_img_fp in char_img_fps:
            os.remove(char_img_fp)

        # https://stackoverflow.com/questions/42262198
        h_, w_ = 32, 100
        for index, (box_list_,
                    score_list_) in enumerate(zip(box_list,
                                                  score_list)):  # noqa
            src_pts = np.array(box_list_.tolist(), dtype=np.float32)
            dst_pts = np.array([[0, 0], [w_, 0], [w_, h_], [0, h_]],
                               dtype=np.float32)
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warp = cv2.warpPerspective(img_origin, M, (w_, h_))
            imageio.imwrite(f"./tmp/reconized/word_{index}.jpg", warp)

    plt.imshow(tmp_img)
    plt.imshow(tmp_pred, cmap='inferno', alpha=args.alpha)
    if args.is_output_polygon:
        img_fn = f"poly_result_{img_fn}"
    else:
        img_fn = f"rect_result_{img_fn}"
    plt.savefig(os.path.join(args.save_dir, img_fn),
                dpi=200,
                bbox_inches='tight')
    gc.collect()
