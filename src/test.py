import os
import gc
import time
import argparse

import torch

from models import DBTextModel
from utils import (device, read_img, test_preprocess, visualize_heatmap,
                   visualize_polygon, str_to_bool)


def load_model(model_path):
    assert os.path.exists(model_path)
    dbnet = DBTextModel().to(device)
    dbnet.load_state_dict(torch.load(model_path, map_location=device))
    return dbnet


def load_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--image_path', type=str, default='./assets/foo.jpg')
    parser.add_argument('--model_path',
                        type=str,
                        default='./models/db_resnet18.pth')
    parser.add_argument('--save_dir', type=str, default='./assets')

    # for heatmap
    parser.add_argument('--prob_thred', type=float, default=0.5)

    # for polygon & rotate rectangle
    parser.add_argument('--heatmap', type=str_to_bool, default=False)
    parser.add_argument('--thresh', type=float, default=0.5)
    parser.add_argument('--box_thresh', type=float, default=0.7)
    parser.add_argument('--unclip_ratio', type=float, default=1.5)
    parser.add_argument('--is_output_polygon', type=str_to_bool, default=False)
    parser.add_argument('--alpha', type=float, default=0.6)

    args = parser.parse_args()
    return args


def main(net, args):
    img_path = args.image_path.replace("file://", "")
    img_fn = img_path.split("/")[-1]
    assert os.path.exists(img_path)
    img_origin, h_origin, w_origin = read_img(img_path)
    tmp_img = test_preprocess(img_origin, to_tensor=True, pad=False).to(device)

    net.eval()
    torch.cuda.empty_cache()
    gc.collect()

    start = time.time()
    with torch.no_grad():
        preds = net(tmp_img)
    print(">>> Inference took {}'s".format(time.time() - start))

    if args.heatmap:
        visualize_heatmap(args, img_fn, tmp_img, preds.to('cpu')[0].numpy())
    else:
        batch = {'shape': [(h_origin, w_origin)]}
        visualize_polygon(args, img_fn, (img_origin, h_origin, w_origin),
                          batch, preds)


if __name__ == '__main__':
    args = load_args()
    dbnet = load_model(args.model_path)

    main(dbnet, args)
