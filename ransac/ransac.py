from sys import argv
import os.path, json
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def generate_data(img_size, line_params, n_points, sigma, inlier_ratio):
    w, h = img_size
    a, b, c = line_params
    assert b != 0
    x = np.random.uniform(low=0, high=w, size=(n_points, 1))
    inliers_count = np.round(n_points * inlier_ratio).astype(np.int)
    y_line = (-a * x[:inliers_count] - c) / b + np.random.normal(0, sigma, (inliers_count, 1))
    y_uniform = np.random.uniform(low=0, high=h, size=(n_points - inliers_count, 1))
    y = np.append(y_line, y_uniform).reshape(x.shape)
    return np.hstack([x, y])


def compute_ransac_thresh(alpha, sigma):
    return stats.chi2.ppf(alpha, sigma)


def compute_ransac_iter_count(conv_prob, inlier_ratio):
    return np.ceil(np.log(1 - conv_prob) / np.log(1 - inlier_ratio ** 2)).astype(np.int)


def compute_line_ransac(data, t, n):
    cost = 0
    best_params = None
    w, h = data.shape
    for i in range(n):
        sample = data[np.squeeze(np.random.randint(0, w, size=(2, 1)))]
        x1, y1 = sample[0]
        x2, y2 = sample[1]
        params = np.array((y1-y2, x2-x1, x1*y2 - x2*y1))
        a, b, c = params
        if b == 0:
            continue
        H = (-a*data[:, 0] - c) / b
        dist = np.abs(H - data[:, 1])
        current_cost = len([1 for x in dist if x <= t])
        if current_cost > cost:
            cost = current_cost
            best_params = params
    if best_params[0] != 0:
        best_params /= best_params[0]
    print("Cost ", cost, " / ", n)
    return best_params


def main():
    print(argv)
    assert len(argv) == 2
    assert os.path.exists(argv[1])

    with open(argv[1]) as fin:
        params = json.load(fin)

    """
    params:
    line_params: (a,b,c) - line params (ax+by+c=0)
    img_size: (w, h) - size of the image
    n_points: count of points to be used
    sigma - Gaussian noise
    alpha - probability of point is an inlier
    inlier_ratio - ratio of inliers in the data
    conv_prob - probability of convergence
    """

    data = generate_data((params['w'], params['h']),
                         (params['a'], params['b'], params['c']),
                         params['n_points'], params['sigma'],
                         params['inlier_ratio'])

    t = compute_ransac_thresh(params['alpha'], params['sigma'])
    n = compute_ransac_iter_count(params['conv_prob'], params['inlier_ratio'])

    detected_line = compute_line_ransac(data, t, n)
    print(detected_line)

    a, b, c = detected_line
    plt.figure(figsize=(10, 60))
    H = (-a * data[:, 0] - c) / b
    plt.scatter(data[:, 0], data[:, 1], s=20, c='r')
    plt.plot(data[:, 0], H, "--", c='blue', alpha=0.7)
    plt.show()


if __name__ == '__main__':
    main()
