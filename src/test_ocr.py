import sys
sys.path.insert(0, "/home/phan.huy.hoang/phh_workspace/")  # noqa
import os
import gc
import time
import glob
import string
import imageio
import argparse

import cv2
import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F

from models import DBTextModel
from utils import (read_img, test_preprocess, visualize_heatmap,
                   visualize_polygon, str_to_bool, draw_bbox)
from postprocess import SegDetectorRepresenter

from clova_ocr.utils import CTCLabelConverter, AttnLabelConverter
from clova_ocr.dataset import test_preprocess as rec_preprocess
from clova_ocr.model import Model


class WrappedModel(nn.Module):
    """convert DataParallel to cpu
    https://discuss.pytorch.org/t/loading-weights-from-dataparallel-models/20570/2
    """
    def __init__(self, module):
        super(WrappedModel, self).__init__()
        self.module = module

    def forward(self, x, y, is_train=False):
        return self.module(x, y, is_train=is_train)


def load_rec_model(opt):
    model = Model(opt)
    model = WrappedModel(model).to(opt.device)

    # load model
    print(">>> loading pretrained model from {}".format(opt.saved_model))
    state_dict = torch.load(opt.saved_model, map_location=opt.device)
    model.load_state_dict(state_dict, strict=False)
    return model


def load_det_model(opt):
    assert os.path.exists(opt.det_model_path)
    dbnet = DBTextModel().to(opt.device)
    dbnet.load_state_dict(
        torch.load(opt.det_model_path, map_location=opt.device))
    return dbnet


def predict(image_tensors, converter, model, opt):

    # predict
    model.eval()
    with torch.no_grad():
        batch_size = image_tensors.size(0)
        image = image_tensors.to(opt.device)
        # For max length prediction
        length_for_pred = torch.IntTensor([opt.batch_max_length] *
                                          batch_size).to(opt.device)
        text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length +
                                         1).fill_(0).to(opt.device)

        preds_str = ''
        if 'CTC' in opt.Prediction:
            preds = model(image, text_for_pred)

            # Select max probabilty (greedy decoding) then decode index to character
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            _, preds_index = preds.max(2)
            preds_index = preds_index.view(-1)
            preds_str = converter.decode(preds_index.data, preds_size.data)
        else:
            start = time.time()
            preds = model(image, text_for_pred, is_train=False)
            print(">>> Recognize: {}".format(time.time() - start))

            # select max probabilty (greedy decoding) then decode index to character
            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index, length_for_pred)

        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)
        if opt.debug:
            print(preds_str, preds_prob.shape, preds_max_prob.shape)

        pred = preds_str[0]
        pred_max_prob = preds_max_prob[0]
        if 'Attn' in opt.Prediction:
            pred_EOS = pred.find('[s]')
            pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
            pred_max_prob = pred_max_prob[:pred_EOS]

        # calculate confidence score (= multiply of pred_max_prob)
        confidence_score = pred_max_prob.cumprod(dim=0)[-1]

        result = {"pred": pred, "score": float(confidence_score)}
        if opt.debug:
            print(result)
        return result


def main(opt, dbnet):

    print(">>> Devide: {}".format(opt.device))

    # TEXT DETECTION
    # (x, y, label)
    if opt.img_path:
        assert os.path.exists(opt.img_path)

    img_path = opt.img_path.replace("file://", "")
    img_fn = img_path.split("/")[-1]
    assert os.path.exists(img_path)
    img_origin, h_origin, w_origin = read_img(img_path)
    tmp_img = test_preprocess(img_origin, to_tensor=True,
                              pad=False).to(opt.device)

    dbnet.eval()
    torch.cuda.empty_cache()
    gc.collect()
    start = time.time()
    with torch.no_grad():
        preds = dbnet(tmp_img)
    print(">>> Detect: {}'s".format(time.time() - start))

    seg_obj = SegDetectorRepresenter(thresh=opt.thresh,
                                     box_thresh=opt.box_thresh,
                                     unclip_ratio=opt.unclip_ratio)
    batch = {'shape': [(h_origin, w_origin)]}
    box_list, score_list = seg_obj(batch,
                                   preds,
                                   is_output_polygon=opt.is_output_polygon)
    box_list, score_list = box_list[0], score_list[0]

    if len(box_list) > 0:
        if opt.is_output_polygon:
            idx = [x.sum() > 0 for x in box_list]
            box_list = [box_list[i] for i, v in enumerate(idx) if v]
            score_list = [score_list[i] for i, v in enumerate(idx) if v]
        else:
            idx = box_list.reshape(box_list.shape[0], -1).sum(axis=1) > 0
            box_list, score_list = box_list[idx], score_list[idx]
    else:
        box_list, score_list = [], []
    img_origin = draw_bbox(img_origin,
                           np.array(box_list),
                           color=(0, 0, 255),
                           thickness=1)

    # https://stackoverflow.com/questions/42262198
    img_warps = []
    h_, w_ = 32, 100
    if not opt.is_output_polygon:

        char_img_fps = glob.glob(os.path.join("./tmp/reconized", "*"))
        for char_img_fp in char_img_fps:
            os.remove(char_img_fp)

        for index, (box_list_,
                    score_list_) in enumerate(zip(box_list,
                                                  score_list)):  # noqa
            src_pts = np.array(box_list_.tolist(), dtype=np.float32)
            dst_pts = np.array([[0, 0], [w_, 0], [w_, h_], [0, h_]],
                               dtype=np.float32)
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warp = cv2.warpPerspective(img_origin, M, (w_, h_))
            imageio.imwrite("./tmp/reconized/word_{}.jpg".format(index), warp)
            img_warps.append((box_list_.tolist()[0], warp))

    # TEXT RECOGNITION
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character, opt.device)
    else:
        converter = AttnLabelConverter(opt.character, opt.device)
    opt.num_class = len(converter.character)
    print(">>> no class: {}".format(opt.num_class))

    if opt.rgb:
        opt.input_channel = 3

    rec_model = Model(opt)
    rec_model = WrappedModel(rec_model).to(opt.device)

    print(">>> loading pretrained model from {}".format(opt.saved_model))
    state_dict = torch.load(opt.saved_model, map_location=opt.device)
    rec_model.load_state_dict(state_dict, strict=False)

    img_origin = img_origin.astype(np.float32)
    for coord, img_warp in img_warps:
        image_tensors = rec_preprocess(img_warp)
        result = predict(image_tensors, converter, rec_model, opt)
        x, y = coord[0], coord[1]
        cv2.circle(img_origin, (x, y),
                   radius=0,
                   color=(0, 255, 0),
                   thickness=int(h_origin * 0.01))
        cv2.putText(img_origin, result['pred'], (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, 255, 2)

    if opt.out_path:
        imageio.imwrite(opt.out_path, img_origin.astype(np.uint8))


def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--webcam', required=False)
    parser.add_argument('--img_path',
                        required=False,
                        help='path to test image')
    parser.add_argument('--out_path', required=False)
    parser.add_argument('--debug', action='store_true', help='just for debug')
    parser.add_argument('--is_output_polygon', action='store_true', help='')
    parser.add_argument('--workers',
                        type=int,
                        help='number of data loading workers',
                        default=1)
    parser.add_argument('--device', type=str, help='cpu/cuda', default='cpu')
    parser.add_argument('--batch_size',
                        type=int,
                        default=1,
                        help='input batch size')
    parser.add_argument('--saved_model',
                        required=True,
                        help="path to saved_model to evaluation")
    parser.add_argument('--det_model_path',
                        type=str,
                        default='./models/db_resnet18.pth')
    """ Data processing """
    # for polygon & rotate rectangle
    parser.add_argument('--thresh', type=float, default=0.25)
    parser.add_argument('--box_thresh', type=float, default=0.50)
    parser.add_argument('--unclip_ratio', type=float, default=1.5)
    parser.add_argument('--alpha', type=float, default=0.6)
    parser.add_argument('--batch_max_length',
                        type=int,
                        default=25,
                        help='maximum-label-length')
    parser.add_argument('--imgH',
                        type=int,
                        default=32,
                        help='the height of the input image')
    parser.add_argument('--imgW',
                        type=int,
                        default=100,
                        help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character',
                        type=str,
                        default='0123456789abcdefghijklmnopqrstuvwxyz',
                        help='character label')
    parser.add_argument('--sensitive',
                        action='store_true',
                        help='for sensitive character mode')
    parser.add_argument('--PAD',
                        action='store_true',
                        help='whether to keep ratio then pad for image resize')
    """ Model Architecture """
    parser.add_argument('--Transformation',
                        type=str,
                        required=True,
                        help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction',
                        type=str,
                        required=True,
                        help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling',
                        type=str,
                        required=True,
                        help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction',
                        type=str,
                        required=True,
                        help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial',
                        type=int,
                        default=20,
                        help='number of fiducial points of TPS-STN')
    parser.add_argument(
        '--input_channel',
        type=int,
        default=1,
        help='the number of input channel of Feature extractor')
    parser.add_argument(
        '--output_channel',
        type=int,
        default=512,
        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size',
                        type=int,
                        default=256,
                        help='the size of the LSTM hidden state')

    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = load_args()
    if opt.sensitive:
        opt.character = string.printable[:-6]

    dbnet = load_det_model(opt)
    # net = load_model(opt)
    main(opt, dbnet)
