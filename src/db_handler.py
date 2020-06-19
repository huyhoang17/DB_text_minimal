import os
import io
import imageio
import logging

import cv2
import numpy as np
from PIL import Image
import torch
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)


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


class DBTextDetectionHandler(BaseHandler):
    def __init__(self):
        self.model = None
        self.device = None
        self.initialized = False

    def initialize(self, ctx):
        self.manifest = ctx.manifest
        properties = ctx.system_properties
        self.device = torch.device('cpu')
        model_dir = properties.get("model_dir")
        serialized_file = self.manifest['model']['serializedFile']
        model_pt_path = os.path.join(model_dir, serialized_file)
        assert os.path.exists(model_pt_path)
        self.model = torch.jit.load(model_pt_path)
        self.model.to(self.device)
        self.model.eval()

        logger.debug(
            'Model file {0} loaded successfully'.format(model_pt_path))
        self.initialized = True

    def preprocess(self, request):

        tensor_imgs = []
        for _, data in enumerate(request):
            image = data.get("data")
            if image is None:
                image = data.get("body")

            input_image = Image.open(io.BytesIO(image))
            input_image = np.array(input_image)
            tensor_img = test_preprocess(input_image, pad=False)
            tensor_imgs.append(tensor_img)

        tensor_imgs = torch.cat(tensor_imgs)
        return tensor_imgs

    def inference(self, img):
        return self.model(img)

    def postprocess(self, data):

        res = []
        data = data.detach().cpu().numpy()
        for pred in data:
            prob_mask = (pred[0] * 255).astype(np.uint8)
            thresh_mask = (pred[1] * 255).astype(np.uint8)
            prob_mask = prob_mask.tolist()
            thresh_mask = thresh_mask.tolist()
            res.append({"prob_mask": prob_mask, "thresh_mask": thresh_mask})

        return res


_service = DBTextDetectionHandler()


def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    data = _service.preprocess(data)
    data = _service.inference(data)
    data = _service.postprocess(data)

    return data
