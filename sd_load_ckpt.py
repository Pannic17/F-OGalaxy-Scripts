import time

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from diffusers import DiffusionPipeline
import matplotlib.pyplot as plt

# model_path = './models/400shijing.safetensors'


checkpoint_path = "./stable-diffusion-v1-5"
pipeline = StableDiffusionPipeline.from_single_file("models/SDv1.5-stellar.ckpt", torch_dtype=torch.float16)
pipeline.enable_attention_slicing()
pipeline = pipeline.to("cuda")

# pipeline.load_lora_weights('./models', weight_name="400shijing.safetensors")

prompt = "A galaxy with fabulous nebula, with stars and planet at near, netron star and black hole far away. animated, hdr, cinematic, illustration"
# pipe.enable_attention_slicing()
image = pipeline(prompt, width=1024, height=1024).images[0]

id = str(int(time.time()))
image.save("output/universe/sd_nasa_{}.png".format(id))
plt.imshow(image)
plt.axis('off')  # No axes for this image
plt.show()
