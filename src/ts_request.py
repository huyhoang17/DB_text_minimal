import os
import json
import time
import imageio
import argparse

import requests
import numpy as np


def load_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--image_path', type=str, default='./assets/foo.jpg')
    parser.add_argument('--model_name', type=str, default='dbtext')
    parser.add_argument('--mode', type=str, default='predictions')

    parser.add_argument('--host', type=str, default='localhost')
    parser.add_argument('--port', type=str, default='8080')

    args = parser.parse_args()
    return args


def main(args):
    url = "http://{}:{}/{}/{}".format(args.host, args.port, args.mode,
                                      args.model_name)
    image_path = args.image_path
    with open(image_path, "rb") as f:
        data = f.read()

    start = time.time()
    resp = requests.post(url, data=data).text
    print("REST took: {}'s".format(time.time() - start))
    resp = json.loads(resp)
    prob_mask = np.array(resp["prob_mask"]).astype(np.uint8)
    thresh_mask = np.array(resp["thresh_mask"]).astype(np.uint8)
    print(prob_mask.shape, thresh_mask.shape)
    imageio.imwrite("./tmp/foo1.jpg", prob_mask)
    imageio.imwrite("./tmp/foo2.jpg", thresh_mask)


if __name__ == '__main__':
    args = load_args()
    main(args)
