from _sha3 import sha3_224

import numpy as np
import matplotlib.pyplot as plt
import cv2
from sys import argv


def gradient_img(img):
    hor_grad = (img[1:, :] - img[:-1, :])[:, :-1]
    ver_grad = (img[:, 1:] - img[:, :-1])[:-1:, :]
    magnitude = np.sqrt(hor_grad ** 2 + ver_grad ** 2)

    return magnitude


def hough_transform(img, theta, rho):
    img /= np.max(img)
    w, h = img.shape
    thetas = np.arange(0, np.pi, theta)
    R = np.sqrt(w**2 + h**2)
    rhos = np.arange(-R, R, rho)

    ht_map = np.zeros(shape=(len(thetas), len(rhos)))
    for i in range(w):
        for j in range(h):
            for k in range(len(thetas)):
                t = thetas[k]
                if img[i, j] > 0:
                    r = i * np.sin(t) + j * np.cos(t)
                    index = np.int16((R + r) / rho)
                    ht_map[k, index] += 1

    return ht_map, thetas, rhos


def get_lines(ht_map, thetas, rhos, n_lines, min_delta_rho, min_delta_theta):
    lines = []
    extremums = []
    for i in range(n_lines):
        extremums += [[0, 0, 0]]

    for k in range(n_lines):
        for i in range(len(thetas)):
            for j in range(len(rhos)):
                flag = True
                for index in range(k):
                    if abs(thetas[extremums[index][1]] - thetas[i]) < min_delta_theta or \
                            abs(rhos[extremums[index][2]] - rhos[j]) < min_delta_rho:
                        flag = False
                if extremums[k][0] < ht_map[i][j] and flag:
                    extremums[k] = [ht_map[i][j], i, j]
    for i in range(n_lines):
        lines += [(-np.sin(thetas[extremums[i][1]]) / np.cos(thetas[extremums[i][1]]),
                   rhos[extremums[k][2]] / np.cos(thetas[extremums[i][1]]))]
    return lines


if __name__ == '__main__':
    assert len(argv) == 9
    src_path, dst_ht_path, dst_lines_path, theta, rho,\
        n_lines, min_delta_rho, min_delta_theta = argv[1:]

    theta = float(theta)
    rho = float(rho)
    n_lines = int(n_lines)
    min_delta_rho = float(min_delta_rho)
    min_delta_theta = float(min_delta_theta)

    assert theta > 0.0
    assert rho > 0.0
    assert n_lines > 0
    assert min_delta_rho > 0.0
    assert min_delta_theta > 0.0

    image = cv2.imread(src_path, 0)
    assert image is not None

    image = image.astype(float)
    gradient = gradient_img(image)

    ht_map, thetas, rhos = hough_transform(gradient, theta, rho)
    cv2.imwrite(dst_ht_path, ht_map)

    lines = get_lines(ht_map, thetas, rhos, n_lines, min_delta_rho, min_delta_theta)
    with open(dst_lines_path, 'w') as fout:
        for line in lines:
            fout.write('%0.3f, %0.3f\n' % line)