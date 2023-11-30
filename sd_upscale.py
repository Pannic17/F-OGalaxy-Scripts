import time

from diffusers import StableDiffusionUpscalePipeline
import torch

import requests
from PIL import Image
from io import BytesIO

# load model and scheduler
model_id = "stabilityai/stable-diffusion-x4-upscaler"
pipeline = StableDiffusionUpscalePipeline.from_pretrained(
    model_id, revision="fp16", torch_dtype=torch.float16
)
pipeline = pipeline.to("cuda")

# let's download an  image
# url = "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd2-upscale/low_res_cat.png"
# response = requests.get(url)
# low_res_img = Image.open(BytesIO(response.content)).convert("RGB")
# low_res_img = low_res_img.resize((128, 128))
low_res_img = Image.open("output/vignette/sd_inpaint_1700378796.png")
prompt = "galaxy"

upscaled_image = pipeline(prompt=prompt, image=low_res_img, num_inference_steps=20).images[0]
# upscaled_image.save("upsampled_cat.png")

# upscaled_image.save("../images/a2.png")

# upscaled_image = pipeline(prompt=prompt, image=image).images[0]
id = str(int(time.time()))
upscaled_image.save("output/upscale/sd_2x_{}.png".format(id))