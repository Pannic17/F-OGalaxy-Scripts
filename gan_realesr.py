import time

import torch
from PIL import Image
from RealESRGAN import RealESRGAN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

start = time.time()
model = RealESRGAN(device, scale=4)
model.load_weights('models/RealESRGAN_x4plus.pth', download=True)

path_to_image = 'output/vignette/sd_inpaint_1700378796.png'
image = Image.open(path_to_image).convert('RGB')

sr_image = model.predict(image)

id = str(int(time.time()))
sr_image.save("output/upscale/gan_4x_{}.png".format(id))
print("Time taken: ", time.time() - start)


def gan_upscale_4x(image_path: str, output_path: str, default_path=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RealESRGAN(device, scale=4)
    model.load_weights('models/RealESRGAN_x4plus.pth', download=True)
    image = Image.open(image_path).convert('RGB')

    sr_image = model.predict(image)

    id = str(int(time.time()))
    if default_path:
        sr_image.save("output/upscale/gan_4x_{}.png".format(id))
    # sr_image = model.predict(image)
    else:
        sr_image.save(output_path)
    return sr_image
