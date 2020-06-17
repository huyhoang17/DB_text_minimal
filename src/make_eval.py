import os
import gc
import sys
import glob
import argparse

from tqdm import tqdm
import joblib
import torch

from models import DBTextModel
from utils import (read_img, test_preprocess, visualize_heatmap,
                   visualize_polygon, str_to_bool)
from postprocess import SegDetectorRepresenter


def load_model(model_path, device):
    assert os.path.exists(model_path)
    dbnet = DBTextModel().to(device)
    dbnet.load_state_dict(torch.load(model_path, map_location=device))
    return dbnet


def load_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--image_dir', type=str)
    parser.add_argument('--model_path',
                        type=str,
                        default='./models/db_resnet18.pth')
    parser.add_argument('--save_dir', type=str, default='./assets')
    parser.add_argument('--device', type=str, default='cuda')

    # for polygon & rotate rectangle
    parser.add_argument('--thresh', type=float, default=0.3)
    parser.add_argument('--box_thresh', type=float, default=0.5)
    parser.add_argument('--unclip_ratio', type=float, default=1.5)
    parser.add_argument('--is_output_polygon', type=str_to_bool, default=True)

    # output
    parser.add_argument('--preds_fp',
                        type=str,
                        default='./data/result_poly_preds.pkl')
    parser.add_argument('--img_fns_fp', type=str, default='./data/img_fns.pkl')

    args = parser.parse_args()
    return args


def to_list_tuples(sample):
    sample = [tuple(i) for i in sample]
    return sample


def main(args):
    """
    Eval for totaltext dataset
    """

    dbnet = load_model(args.model_path, args.device)

    test_img_fps = sorted(glob.glob(os.path.join(args.image_dir, "*")))

    result_poly_preds = []
    img_fns = []
    for test_img_fp in tqdm(test_img_fps):
        try:
            test_img_fn = test_img_fp.split("/")[-1]
            img_fns.append(test_img_fn)
            img_origin, h_origin, w_origin = read_img(test_img_fp)
            tmp_img = test_preprocess(img_origin)

            tmp_img = tmp_img.to(args.device)
            batch = {'shape': [(h_origin, w_origin)]}

            with torch.no_grad():
                preds = dbnet(tmp_img)
            torch.cuda.empty_cache()

            seg_obj = SegDetectorRepresenter(thresh=args.thresh,
                                             box_thresh=args.box_thresh,
                                             unclip_ratio=args.unclip_ratio)
            box_list, score_list = seg_obj(
                batch, preds, is_output_polygon=args.is_output_polygon)
            box_list, score_list = box_list[0], score_list[0]

            if len(box_list) > 0:
                if args.is_output_polygon:
                    idx = [x.sum() > 0 for x in box_list]
                    box_list = [box_list[i] for i, v in enumerate(idx) if v]
                    score_list = [
                        score_list[i] for i, v in enumerate(idx) if v
                    ]
                else:
                    idx = box_list.reshape(box_list.shape[0],
                                           -1).sum(axis=1) > 0
                    box_list, score_list = box_list[idx], score_list[idx]
            else:
                box_list, score_list = [], []

            preds_per_img = []
            for poly_pred in box_list:
                poly_pred = to_list_tuples(poly_pred)
                pred_sample = {
                    # polygon, list of point coordinates
                    "points": poly_pred,
                    "text": "text_sample",
                    "ignore": False
                }
                preds_per_img.append(pred_sample)
            result_poly_preds.append(preds_per_img)
            torch.cuda.empty_cache()
            gc.collect()

        except Exception:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno, test_img_fp)
            result_poly_preds.append([])
            continue

    joblib.dump(result_poly_preds, args.preds_fp)
    joblib.dump(img_fns, args.img_fns_fp)


if __name__ == '__main__':
    args = load_args()
    main(args)
