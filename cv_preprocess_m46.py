import cv2
import os
import numpy as np

img = cv2.imread("D:/Final/OGX/I/messier/m46.webp")
crop_size = min(img.shape[0], img.shape[1]) // 2
center_x, center_y = img.shape[1] - crop_size, img.shape[0] // 2
# Crop the image
cropped_img = img[center_y - crop_size:center_y + crop_size, center_x - crop_size:center_x + crop_size]

# Resize the image to 512x512
resized_img = cv2.resize(cropped_img, (512, 512))

# Save the processed image
cv2.imwrite("D:/Final/OGX/I/train/m46.png", resized_img)