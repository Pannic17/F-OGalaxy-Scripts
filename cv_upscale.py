import time

import numpy as np
from matplotlib import pyplot as plt
from super_image import EdsrModel, ImageLoader, DrlnModel
from PIL import Image
import requests

url = 'https://paperswithcode.com/media/datasets/Set5-0000002728-07a9793f_zA3bDjj.jpg'
# image = Image.open(requests.get(url, stream=True).raw)

image = Image.open("output/universe/sd_moon_1700211817.png")

model = DrlnModel.from_pretrained('eugenesiow/drln-bam', scale=4)
inputs = ImageLoader.load_image(image)
preds = model(inputs)

# ImageLoader.save_image(preds, './scaled_2x.png')

upscaled_image = Image.fromarray(np.uint8(ImageLoader._process_image_to_save(preds)))

id = str(int(time.time()))
upscaled_image.save("output/upscale/cv_2x_{}.png".format(id))
# ImageLoader.save_compare(inputs, preds, './scaled_2x_compare.png')
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(upscaled_image)
plt.axis('off')

plt.show()