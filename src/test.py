import os
import gc
import time
import argparse

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

from models import DBTextModel
from utils import minmax_scaler_img

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"


def test_resize(img, size=600, pad=False):
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


def test_preprocess(img_fp,
                    mean=[103.939, 116.779, 123.68],
                    to_tensor=True,
                    pad=False):
    img = cv2.imread(img_fp)[:, :, ::-1]
    img = test_resize(img, size=600, pad=pad)

    img = img.astype(np.float32)
    img[..., 0] -= mean[0]
    img[..., 1] -= mean[1]
    img[..., 2] -= mean[2]
    img = np.expand_dims(img, axis=0)

    if to_tensor:
        img = torch.Tensor(img.transpose(0, 3, 1, 2))

    return img


def load_model():
    dbnet = DBTextModel().to(device)
    dbnet.load_state_dict(
        torch.load("./models/best_cps.pth", map_location=device),
    )
    return dbnet


def load_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        '--image_path', type=str
    )
    parser.add_argument(
        '--prob_thred', type=float, default=0.3
    )
    args = parser.parse_args()
    return args


def main(net, args):
    img_path = args.image_path.replace("file://", "")
    img_fn = img_path.split("/")[-1]
    assert os.path.exists(img_path)
    tmp_img = test_preprocess(
        img_path, to_tensor=True, pad=True
    ).to(device)

    net.eval()
    torch.cuda.empty_cache()
    gc.collect()

    start = time.time()
    with torch.no_grad():
        tmp_pred = net(tmp_img).to('cpu')[0].numpy()
    print(tmp_pred.shape)
    print(">>> Inference took {}'s".format(time.time() - start))

    pred_prob = tmp_pred[0]
    pred_prob[pred_prob <= args.prob_thred] = 0
    pred_prob[pred_prob > args.prob_thred] = 1

    np_img = minmax_scaler_img(tmp_img[0].to(device).numpy().transpose((1, 2, 0)))  # noqa
    plt.imshow(np_img)
    plt.imshow(pred_prob, cmap='jet', alpha=0.5)
    plt.savefig('./assets/{}'.format(img_fn), bbox_inches='tight')
    gc.collect()


if __name__ == '__main__':
    dbnet = load_model()
    args = load_args()

    main(dbnet, args)
