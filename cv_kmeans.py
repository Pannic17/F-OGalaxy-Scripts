import cv2
import numpy as np
from matplotlib import pyplot as plt


def cluster_generate_color(image, k: int = 5):
    # image = self.generated
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    return centers


image = cv2.imread("output/universe/sd_moon_1700211817.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
centers = cluster_generate_color(image, k=5)
plt.figure(figsize=(8, 6))
for i, color in enumerate(centers):
    print(color)
    plt.subplot(1, 5, i + 1)
    plt.axis('off')
    plt.title(f'Color {i + 1}')
    plt.imshow([[color]])

plt.show()
