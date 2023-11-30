import cv2
import numpy as np
from matplotlib import pyplot as plt

# 读取图片
original_image = cv2.imread("output/universe/sd_nasa_1700211555.png")
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
t_height, t_width = 1024, 576

canvas = np.zeros((t_height, t_width, 3), dtype=np.uint8)
x_offset = (t_width - original_image.shape[1]) // 2
y_offset = (t_height - original_image.shape[0]) // 2
canvas[y_offset:y_offset + original_image.shape[0], x_offset:x_offset + original_image.shape[1]] = original_image

# 创建暗角掩码
x = np.arange(t_width, dtype=np.float32) - t_width / 2
y = np.arange(t_height, dtype=np.float32) - t_height / 2
x, y = np.meshgrid(x, y)
distance_from_center = np.sqrt(x ** 2 + y ** 2)

center_fraction = 0.8
center_radius = min(t_height, t_width) / 2 * center_fraction
radius = min(t_height, t_width) / 2
vignette_center_mask = np.clip((distance_from_center - center_radius) / (radius - center_radius), 0, 1)
vignette_center = cv2.merge([vignette_center_mask, vignette_center_mask, vignette_center_mask])

vignette_image = vignette_center * canvas

# 显示原始图片和带暗角的图片
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Vignette Effect")
plt.imshow(cv2.cvtColor(vignette_image.astype(np.uint8), cv2.COLOR_BGR2RGB))
plt.axis('off')

white = np.ones((t_height, t_width, 3), dtype=np.uint8) * 255
vignette_mask = white * vignette_center

plt.subplot(1, 3, 3)
plt.title("Vignette Mask")
plt.imshow(cv2.cvtColor(vignette_mask.astype(np.uint8), cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.show()


# 应用新的暗角掩码到图片上
# vignette_center_clear_image = vignette_center_mask * canvas

def create_target_image(image, width, height):
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x_offset = (width - original_image.shape[1]) // 2
    y_offset = (height - original_image.shape[0]) // 2
    canvas[y_offset:y_offset + original_image.shape[0], x_offset:x_offset + original_image.shape[1]] = original_image
    return canvas


def add_vignette(image, width, height, center_fraction=0.9):
    x = np.arange(width, dtype=np.float32) - width / 2
    y = np.arange(height, dtype=np.float32) - height / 2
    x, y = np.meshgrid(x, y)
    distance_from_center = np.sqrt(x ** 2 + y ** 2)

    center_radius = min(height, width) / 2 * center_fraction
    radius = min(height, width) / 2
    vignette_mask = np.clip(1- (distance_from_center - center_radius) / (radius - center_radius), 0, 1)
    vignette_mask = cv2.merge([vignette_mask, vignette_mask, vignette_mask])
    vignette_image = image * vignette_mask
    return vignette_image


def create_vignette_mask(width, height, center_fraction=0.8):
    center_radius = min(height, width) / 2 * center_fraction
    vignette_center = np.ones((height, width, 3), dtype=np.uint8) * 255
    center_x, center_y = width // 2, height // 2
    cv2.circle(vignette_center, (center_x, center_y), int(center_radius), (0, 0, 0), -1)
    return vignette_center


def extend_image_horizontal(original_image, width, height):
    extend_image = np.zeros((height, width, 3), dtype=np.uint8)

    # Extract the first and last column of pixels from the original image
    first_column = original_image[:, 0, :]
    last_column = original_image[:, -1, :]

    # Calculate the left and right padding sizes
    left_padding = (width - original_image.shape[1]) // 2
    right_padding = width - original_image.shape[1] - left_padding

    for i in range(left_padding):
        alpha = i / left_padding
        extend_image[:, i] = alpha * first_column + (1 - alpha) * np.zeros((3,), dtype=np.uint8)

    # Place the original image in the center
    extend_image[:, left_padding:left_padding + original_image.shape[1]] = original_image

    for i in range(right_padding):
        alpha = i / right_padding
        extend_image[:, -right_padding + i] = (1 - alpha) * last_column + alpha * np.zeros((3,), dtype=np.uint8)

    return extend_image


def extend_image_all(original_image, width, height):
    # Create a black canvas with the new dimensions
    extended_image = np.zeros((height, width, 3), dtype=np.uint8)

    # Calculate the offsets to center the original image
    x_offset = (width - original_image.shape[1]) // 2
    y_offset = (height - original_image.shape[0]) // 2

    # Place the original image in the center of the black canvas
    extended_image[y_offset:y_offset + original_image.shape[0],
    x_offset:x_offset + original_image.shape[1]] = original_image
    return extended_image


def create_extend_canvas(width, height):
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    return canvas


def place_enlarge_blur(canvas, image):
    # enlarge image to the size of canvas and kept the aspect ratio
    target_height = canvas.shape[0]
    # resize image to target height
    target_width = target_height
    # resize image to target width and height
    image = cv2.resize(image, (target_width, target_height))
    # place the iamge in the center of canvas
    x_offset = (canvas.shape[1] - image.shape[1]) // 2
    y_offset = (canvas.shape[0] - image.shape[0]) // 2
    canvas[y_offset:y_offset + image.shape[0], x_offset:x_offset + image.shape[1]] = image
    # canvas[image.shape[0], x_offset:x_offset + image.shape[1]] = image
    # blur the canvas
    canvas = cv2.GaussianBlur(canvas, (51, 51), 0)
    return canvas


def place_center_image(canvas, image):
    # place image in the center of canvas
    x_offset = (canvas.shape[1] - image.shape[1]) // 2
    y_offset = (canvas.shape[0] - image.shape[0]) // 2
    canvas[y_offset:y_offset + image.shape[0], x_offset:x_offset + image.shape[1]] = image
    return canvas
