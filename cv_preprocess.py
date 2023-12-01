import cv2
import os
import numpy as np

def process_image(file_path, output_path):
    # Read the image
    img = cv2.imread(file_path)

    # Compute the center coordinates of the image
    center_x, center_y = img.shape[1] // 2, img.shape[0] // 2

    # Determine the size of the square crop (minimum of width and height)
    crop_size = min(img.shape[0], img.shape[1]) // 2

    # Crop the image
    cropped_img = img[center_y - crop_size:center_y + crop_size, center_x - crop_size:center_x + crop_size]

    # Resize the image to 512x512
    resized_img = cv2.resize(cropped_img, (512, 512))

    # Save the processed image
    cv2.imwrite(output_path, resized_img)

# Folder containing the images
input_folder = 'D:/Final/OGX/I/messier'
output_folder = 'D:/Final/OGX/I/train'

# Ensure output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Process each image in the folder
for file_name in os.listdir(input_folder):
    file_path = os.path.join(input_folder, file_name)
    # output_path = os.path.join(output_folder, file_name)
    output_path = os.path.join(output_folder, file_name.split('.')[0] + '.png')
    process_image(file_path, output_path)
    # Check if the file is an image
    # if os.path.isfile(file_path) and file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
    #     process_image(file_path, output_path)

print("Processing complete.")
