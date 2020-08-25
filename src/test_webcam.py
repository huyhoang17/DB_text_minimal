import os
import gc
import sys
sys.path.insert(0, "/home/phan.huy.hoang/phh_workspace/")  # noqa
import time
import copy
import argparse

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import DBTextModel
from utils import (read_img, test_preprocess, visualize_heatmap,
                   visualize_polygon, str_to_bool, draw_bbox, timer)
from postprocess import SegDetectorRepresenter

from clova_ocr.utils import CTCLabelConverter, AttnLabelConverter
from clova_ocr.dataset import test_preprocess as rec_preprocess
from clova_ocr.model import Model

os.environ['DISPLAY'] = ':0'


def load_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--recognize', action='store_true')
    parser.add_argument('--show_video', action='store_true')
    parser.add_argument('--device', type=str, help='cpu/cuda', default='cuda')
    parser.add_argument('--workers',
                        type=int,
                        help='number of data loading workers',
                        default=1)
    parser.add_argument('--batch_size',
                        type=int,
                        default=1,
                        help='input batch size')
    parser.add_argument('--saved_model', type=str)
    parser.add_argument('--det_model_path', type=str)
    parser.add_argument('--save_dir', type=str, default='./assets')
    parser.add_argument('--video_path', type=str, default='')
    parser.add_argument('--per_frame', type=int, default=5)

    # for heatmap
    parser.add_argument('--prob_thred', type=float, default=0.5)

    # for polygon & rotate rectangle
    parser.add_argument('--heatmap', type=str_to_bool, default=False)
    parser.add_argument('--thresh', type=float, default=0.30)
    parser.add_argument('--box_thresh', type=float, default=0.62)
    parser.add_argument('--unclip_ratio', type=float, default=1.5)
    parser.add_argument('--alpha', type=float, default=0.6)
    parser.add_argument('--num_class', type=int, default=38)
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

    args = parser.parse_args()
    return args


class WrappedModel(nn.Module):
    """convert DataParallel to cpu
    https://discuss.pytorch.org/t/loading-weights-from-dataparallel-models/20570/2
    """
    def __init__(self, module):
        super(WrappedModel, self).__init__()
        self.module = module

    def forward(self, x, y, is_train=False):
        return self.module(x, y, is_train=is_train)


def load_det_model(args):
    assert os.path.exists(args.det_model_path)
    dbnet = DBTextModel().to(args.device)
    dbnet.load_state_dict(
        torch.load(args.det_model_path, map_location=args.device))
    dbnet.eval()
    return dbnet


def load_rec_model(args):
    rec_model = Model(args)
    rec_model = WrappedModel(rec_model).to(args.device)
    state_dict = torch.load(args.saved_model, map_location=args.device)
    rec_model.load_state_dict(state_dict, strict=False)
    rec_model.eval()

    return rec_model


@timer
def predict(image_tensors, converter, model, args):

    # predict
    with torch.no_grad():
        batch_size = image_tensors.size(0)
        image = image_tensors.to(args.device)
        # For max length prediction
        length_for_pred = torch.IntTensor([args.batch_max_length] *
                                          batch_size).to(args.device)
        text_for_pred = torch.LongTensor(batch_size, args.batch_max_length +
                                         1).fill_(0).to(args.device)

        preds_str = ''
        if 'CTC' in args.Prediction:
            preds = model(image, text_for_pred)

            # Select max probabilty (greedy decoding) then decode index to character
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            _, preds_index = preds.max(2)
            preds_index = preds_index.view(-1)
            preds_str = converter.decode(preds_index.data, preds_size.data)
        else:
            start = time.time()
            preds = model(image, text_for_pred, is_train=False)
            # print(">>> Recognize: {}".format(time.time() - start))

            # select max probabilty (greedy decoding) then decode index to character
            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index, length_for_pred)

        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)

        pred = preds_str[0]
        pred_max_prob = preds_max_prob[0]
        if 'Attn' in args.Prediction:
            pred_EOS = pred.find('[s]')
            pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
            pred_max_prob = pred_max_prob[:pred_EOS]

        # calculate confidence score (= multiply of pred_max_prob)
        confidence_score = pred_max_prob.cumprod(dim=0)[-1]

        result = {"pred": pred, "score": float(confidence_score)}
        return result


def main(det_model, rec_model, args):

    if args.video_path:
        cap = cv2.VideoCapture(args.video_path)
    else:
        cap = cv2.VideoCapture(0)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(fps)
    count = 0
    per_frame = args.per_frame

    # save video
    if args.video_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('./tmp/out.mp4', fourcc, 20.0, (640, 480))

    # detection process
    seg_obj = SegDetectorRepresenter(thresh=args.thresh,
                                     box_thresh=args.box_thresh,
                                     unclip_ratio=args.unclip_ratio)

    # recognizion process
    if 'CTC' in args.Prediction:
        converter = CTCLabelConverter(args.character, args.device)
    else:
        converter = AttnLabelConverter(args.character, args.device)
    args.num_class = len(converter.character)

    if args.rgb:
        args.input_channel = 3

    while True:
        ret, frame = cap.read()
        # default shape = (480, 640, 3), uint8
        if args.video_path:
            frame = cv2.resize(frame, (640, 480))

        if count % per_frame == 0:
            h_origin, w_origin, _ = frame.shape

            img = test_preprocess(frame, to_tensor=True,
                                  pad=False).to(args.device)

            # TEXT DETECTION
            start = time.time()
            with torch.no_grad():
                preds = det_model(img)
            end = time.time() - start
            print(">>> Detect: {}'s".format(end))

            batch = {'shape': [(h_origin, w_origin)]}
            box_list, score_list = seg_obj(batch,
                                           preds,
                                           is_output_polygon=False)
            box_list, score_list = box_list[0], score_list[0]
            frame = draw_bbox(frame,
                              np.array(box_list),
                              color=(0, 0, 255),
                              thickness=1)

            if args.recognize:
                # https://stackoverflow.com/questions/42262198
                img_warps = []
                h_, w_ = 32, 100

                for index, (box_list_, score_list_) in enumerate(
                        zip(box_list, score_list)):  # noqa
                    if score_list_ >= args.box_thresh:
                        src_pts = np.array(box_list_.tolist(), dtype=np.float32)
                        dst_pts = np.array([[0, 0], [w_, 0], [w_, h_], [0, h_]],
                                        dtype=np.float32)
                        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                        warp = cv2.warpPerspective(frame, M, (w_, h_))
                        img_warps.append((box_list_.tolist()[0], warp))

                        # TEXT RECOGNITION
                        for coord, img_warp in img_warps:
                            image_tensors = rec_preprocess(img_warp)
                            result = predict(image_tensors, converter, rec_model,
                                            args)
                            x, y = coord[0], coord[1]
                            # cv2.circle(frame, (x, y),
                            #         radius=0,
                            #         color=(0, 255, 0),
                            #         thickness=int(h_origin * 0.01))
                            cv2.putText(frame, result['pred'], (x, y),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, 255, 1)

            if args.show_video:
                cv2.imshow('frame', frame)

            if args.video_path:
                out.write(frame)
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        count += 1

    # When everything done, release the capture
    cap.release()
    if args.video_path:
        out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    args = load_args()

    det_model = load_det_model(args)
    rec_model = load_rec_model(args)

    main(det_model, rec_model, args)
