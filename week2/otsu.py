from sys import argv
import os.path
import numpy as np
import cv2
import matplotlib.pyplot as plt


def params(array, total_count):
    """
    Calculated mean, variance and weight of background/foreground.
    """
    if (len(array) == 0):
        return (0, 0, 0)
    else:
        weight = len(array) / total_count
        mean = np.mean(array)
        std = np.std(array)
        return weight, mean, std

def otsu(src_path, dst_path):
    """
    Returns otsu threshold and saves binary image.
    """
    src_img = cv2.imread(src_path)
    src_img = np.sum(src_img, axis=2) / 3
    n = src_img.shape[0] * src_img.shape[1]

    min_s = 999
    t = None

    for T in range(255):
        background = src_img[src_img <= T]
        foreground = src_img[src_img > T]
        w1, m1, s1 = params(background, n)
        w2, m2, s2 = params(foreground, n)
        s_mutual = w1*s1 + w2*s2
        if s_mutual < min_s:
            min_s = s_mutual
            t = T

    print("Otsu threshold: ", t)

    src_img[src_img < t] = 0
    src_img[src_img >= t] = 255
    cv2.imwrite(dst_path, src_img)
    return t


if __name__ == '__main__':
    assert len(argv) == 3
    assert os.path.exists(argv[1])
    otsu(*argv[1:])