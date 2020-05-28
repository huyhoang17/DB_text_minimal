import numpy as np
import matplotlib.pyplot as plt


def to_device(batch, device='cuda'):
    new_batch = []
    new_batch.append(batch[0])
    for ele in batch[1:]:
        new_batch.append(ele.to(device))
    return new_batch


def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


def minmax_scaler_img(img):
    img = ((img - img.min()) * (1 / (img.max() - img.min()) * 255)).astype(
        'uint8')  # noqa
    return img
