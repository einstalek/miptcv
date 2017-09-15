import cv2
import numpy as np
import matplotlib.pyplot as plt


def gamma_correction(src_path, dest_path, a, b):
    """
    Performs gamma correction.

    :param src_path: path to the source image in [0, 1] range
    :param dest_path: path to new image
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

    plt.imshow(corrected_image, cmap='gray')
    plt.show()
    cv2.imwrite(dest_path + "corrected_image." + ext, corrected_image)

if __name__ == "__main__":
    src = "tst_2.jpg"
    dest = ""
    gamma_correction(src_path=src, dest_path=dest, a=1, b=0.3)