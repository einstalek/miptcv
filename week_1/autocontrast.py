import numpy as np
import os.path
from sys import argv
import matplotlib.pyplot as plt
import cv2

def fix_brightness(x, bright_min, bright_max):
    if x < bright_min:
        x = 0
    elif x > bright_max:
        x = 255
    else:
        x = x * 255 / (bright_max - bright_min)
    return x

def autontrast(src_path, dest_path, white_perc, black_perc):
    """

    :param src_path: Source image path
    :param dest_path: Destination image path
    :param white_perc: Percentage of white pixels
    :param black_perc: Percentage of black pixels
    """
    img = cv2.imread(src_path)
    img = np.sum(img, axis=2) / 3

    # Получаем функцию распределения
    hist, _ = np.histogram(img, bins=256, range=(0, 256))
    cdf = np.cumsum(hist)

    # Выкидываем из распределения белые и черные пиксели
    cdf_n = np.ma.masked_less_equal(cdf, cdf.max()*white_perc)
    cdf_n = np.ma.masked_greater_equal(cdf_n, cdf.max()*(1-black_perc))

    # Получаем минимальныую и максимальную яркости
    bright_min = cdf_n.argmin()
    bright_max = cdf_n.argmax()

    # Применяем функцию автоконтраста к пикселям
    img_flatten = img.flatten()
    img_flatten = np.array(list(map(lambda x: fix_brightness(x, bright_min, bright_max), img_flatten)))
    fixed = img_flatten.reshape(img.shape)

    cv2.imwrite(dest_path, fixed)

if __name__=="__main__":
    assert len(argv) == 5
    assert os.path.exists(argv[1])
    argv[3] = float(argv[3])
    argv[4] = float(argv[4])

    assert 0 <= argv[3] <= 1
    assert 0 <= argv[4] <= 1

    autontrast(*argv[1:])






