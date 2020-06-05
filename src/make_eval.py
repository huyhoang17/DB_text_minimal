import os
import gc
import sys
import glob

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


def to_list_tuples(sample):
    sample = [tuple(i) for i in sample]
    return sample


def main(device='cuda'):

    thresh = 0.3
    box_thresh = 0.70
    unclip_ratio = 3.0
    is_output_polygon = True

    dbnet = load_model("./models/db_resnet18.pth", device)

    TEST_IMG_DIR = "/home/phan.huy.hoang/phh_workspace/data/totaltext/totaltext/Images/Test"
    test_img_fps = sorted(glob.glob(os.path.join(TEST_IMG_DIR, "*")))

    result_poly_preds = []
    img_fns = []
    for test_img_fp in tqdm(test_img_fps):
        try:
            test_img_fn = test_img_fp.split("/")[-1]
            img_fns.append(test_img_fn)
            img_origin, h_origin, w_origin = read_img(test_img_fp)
            tmp_img = test_preprocess(img_origin)

            tmp_img = tmp_img.to(device)
            batch = {'shape': [(h_origin, w_origin)]}

            with torch.no_grad():
                preds = dbnet(tmp_img)
            torch.cuda.empty_cache()

            seg_obj = SegDetectorRepresenter(thresh=thresh,
                                             box_thresh=box_thresh,
                                             unclip_ratio=unclip_ratio)
            box_list, score_list = seg_obj(
                batch, preds, is_output_polygon=is_output_polygon)
            box_list, score_list = box_list[0], score_list[0]

            if len(box_list) > 0:
                if is_output_polygon:
                    idx = [x.sum() > 0 for x in box_list]
                    box_list = [box_list[i] for i, v in enumerate(idx) if v]
                    score_list = [score_list[i]
                                  for i, v in enumerate(idx) if v]
                else:
                    idx = box_list.reshape(
                        box_list.shape[0], -1).sum(axis=1) > 0
                    box_list, score_list = box_list[idx], score_list[idx]
            else:
                box_list, score_list = [], []

    #         print(len(box_list), len(score_list))

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

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno, test_img_fp)
            result_poly_preds.append([])
            continue

    joblib.dump(result_poly_preds, "./data/result_poly_preds.pkl")
    joblib.dump(img_fns, "./data/img_fns.pkl")


if __name__ == '__main__':
    main(device='cuda')
