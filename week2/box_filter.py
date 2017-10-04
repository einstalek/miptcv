import numpy as np
import matplotlib.pyplot as plt
import cv2
from sys import argv
import os.path
from scipy.signal import convolve2d


def box_flter(src_path, dst_path, w, h):
    """
    Saves transformed image.
    """
    src_img = cv2.imread(src_path)
    src_img = src_img.astype(np.float)
    src_img /= 255.

    kernel = np.ones(shape=(w, h)) / (w*h)

    r = convolve2d(src_img[:, :, 0], kernel, mode="same")
    g = convolve2d(src_img[:, :, 1], kernel, mode="same")
    b = convolve2d(src_img[:, :, 2], kernel, mode="same")

    dst_img = np.empty_like(src_img)
    dst_img[:,:,0] = r
    dst_img[:,:,1] = g
    dst_img[:,:,2] = b
    dst_img = (dst_img * 255).astype(np.int)

    cv2.imwrite(dst_path, dst_img)


if __name__ == "__main__":
    assert len(argv) == 5
    assert os.path.exists(argv[1])
    argv[3] = int(argv[3])
    argv[4] = int(argv[4])
    assert argv[3] > 0
    assert argv[4] > 0

    box_flter(*argv[1:])