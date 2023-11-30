# !pip install transformers accelerate
import time

import cv2
from PIL import Image
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, DDIMScheduler
from diffusers.utils import load_image
import numpy as np
import torch
from matplotlib import pyplot as plt

from cv_vignette import create_target_image, add_vignette, create_vignette_mask, extend_image_horizontal, \
    extend_image_all, create_extend_canvas, place_enlarge_blur, place_center_image

# init_image = load_image("https://huggingface.co/datasets/diffusers/test-arrays/resolve/main/stable_diffusion_inpaint/boy.png")
# mask_image = load_image("https://huggingface.co/datasets/diffusers/test-arrays/resolve/main/stable_diffusion_inpaint/boy_mask.png")
width, height = 1280, 720

original_image = cv2.imread("output/universe/sd_nasa_1700211647.png")
# canvas = create_target_image(original_image, width, height)
invert = add_vignette(width, height)
center = create_vignette_mask(width, height, center_fraction=0.7)
# center = invert
# extend = extend_image_horizontal(original_image, width, height)
canvas = create_extend_canvas(width, height)
canvas = place_enlarge_blur(canvas, original_image)


# extend = extend_image_all(original_image, width, height)
# init_image = canvas * invert
extend = place_center_image(canvas, original_image)
init_image = cv2.cvtColor(extend.astype(np.uint8), cv2.COLOR_BGR2RGB)
init_image = Image.fromarray(init_image)
init_image = init_image.resize((width, height))
mask_image = cv2.cvtColor(center.astype(np.uint8), cv2.COLOR_BGR2RGB)
mask_image = Image.fromarray(mask_image)
mask_image = mask_image.resize((width, height))


# generator = torch.Generator(device="cpu").manual_seed(1)


def make_canny_condition(image):
    image = np.array(image)
    image = cv2.Canny(image, 100, 200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    image = Image.fromarray(image)
    return image


control_image = make_canny_condition(init_image)

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_inpaint", torch_dtype=torch.float16, width=width, height=height, use_safetensors=True
)
pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16, safety_checker=None, use_safetensors=True
)

pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

pipe.to("cuda")

# generate image
image = pipe(
    "universe, hdr, nebula, vignette, black on the edge, dark, space, galaxy, stars",
    num_inference_steps=50,
    # generator=generator,
    eta=1.0,
    image=init_image,
    mask_image=mask_image,
    control_image=control_image,
    width=width,
    height=height,
).images[0]

plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.title("init")
plt.imshow(init_image)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("mask")
plt.imshow(mask_image)
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("generated")
plt.imshow(image)
plt.axis('off')
id = str(int(time.time()))
image.save("output/vignette/sd_inpaint_{}.png".format(id))

plt.show()
