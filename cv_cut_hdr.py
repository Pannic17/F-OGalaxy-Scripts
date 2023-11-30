import os

import cv2
# import imageio
import numpy as np

# Path to the HDR image
hdr_image_path = 'input/hdr/Skybox_01.hdr'


# resized_image = cv2.resize(crop_img, (1024, 1024))

# cv2.imshow('hdr_image', image_8bit)
# cv2.waitKey(0)
# Save the cropped image as PNG
# cv2.imwrite('output/train/cropped_image_c.png', resized_image)


def read_hdr(path):
    hdr_image = cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
    # tonemap = cv2.createTonemapReinhard(1.0, 0, 0, 0)
    # ldr = tonemap.process(hdr_image)

    # ldr = cv2.normalize(hdr_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    image_8bit = np.clip(hdr_image * 256, 0, 255).astype('uint8')
    # image_rgb = cv2.cvtColor(image_8bit, cv2.COLOR_BGR2RGB)
    return image_8bit


def crop_image(image):
    height, width, _ = image.shape
    size = height
    sides = (width - size) // 2
    start_x, end_x = sides, width - sides
    start_y, end_y = 0, height
    crop_img = image[start_y:end_y, start_x:end_x]
    return crop_img


def resize_save_image(image, original_path):
    print(original_path)
    resized_image = cv2.resize(image, (1024, 1024))
    folder = original_path.split('\\')[1]
    index = original_path.split('_')[-1].split('.')[0]
    output_path = 'output/train/Skybox_{}_{}.png'.format(folder, index)
    cv2.imwrite(output_path, resized_image)
    print("save image to {}".format(output_path))


main_skybox_path = "D:/Final/OGX/R/ue/Skybox"

for root, dirs, files in os.walk(main_skybox_path):
    for file in files:
        # Check if the file is an HDR file
        if file.endswith('.hdr'):
            # check if the filename does not contain 'alpha' or 'Alpha'
            if 'alpha' not in file.lower():
                # Construct the full file path
                file_path = os.path.join(root, file)
                image = read_hdr(file_path)
                crop_img = crop_image(image)
                resize_save_image(crop_img, file_path)
