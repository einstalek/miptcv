import cv2
import numpy as np
from sys import argv
import os
import matplotlib.pyplot as plt


def gamma_correction(src_path, dest_path, a, b):
    """
    Performs gamma correction.

    :param src_path: path to the source image in [0, 1] range
    :param dest_path: path to a new image
    :param a: parameter of transformation
    :param b: parameter of transformation
    """
    ext = src_path.split(".")[-1]

    img = cv2.imread(src_path)
    img = np.sum(img, axis=2) / 3
    img /= 255
    indexes = img > 1
    img[indexes] = 1

    corrected_image = a * img**b
    indexes = corrected_image > 1
    corrected_image[indexes] = 1
    corrected_image *= 255
    
    cv2.imwrite(dest_path + "corrected_image." + ext, corrected_image)


if __name__ == "__main__":
    assert len(argv) == 5
    assert os.path.exists(argv[1])
    argv[3] = float(argv[3])
    argv[4] = float(argv[4])
    gamma_correction(*argv[1:])
