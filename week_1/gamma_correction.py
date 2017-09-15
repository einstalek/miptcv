import numpy as np
import cv2
import matplotlib.pyplot as plt


def gamma_correction(src, a, b):
    """
    Performs gamma correction.

    :param src: source image in [0, 1] range
    :param a: param of transformation
    :param b: param of transformation
    :return: new image
    """
    new_image = a * src**b
    return new_image

path = 'tst_1.png'
img = cv2.imread(path)

print("Original shape: ", img.shape)
print("Type: ", type(img))

scaled = cv2.resize(img, None, fx=0.5, fy=0.5)
scaled = cv2.cvtColor(scaled, cv2.COLOR_BGR2RGB)
scaled = scaled / 255
print("Scaled image: ", scaled.shape)

gray = np.sum(scaled, axis=2) / 3
gamma_corrected = gamma_correction(gray, 1, 0.5)

plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(gray, cmap='gray')
plt.subplot(122)
plt.imshow(gamma_corrected, cmap='gray')
plt.show()