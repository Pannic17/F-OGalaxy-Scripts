import time

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from diffusers import DiffusionPipeline
import matplotlib.pyplot as plt

model_name = 'greenmoon.safetensors'


checkpoint_path = "./stable-diffusion-v1-5"
pipeline = StableDiffusionPipeline.from_pretrained(checkpoint_path)
pipeline.enable_attention_slicing()
pipeline = pipeline.to("cuda")

pipeline.load_lora_weights('./models', weight_name=model_name)

prompt = "A galaxy contains stars and planets, including cyan netron stars, black holes far away. animated, hdr, cinematic, illustration"
# pipe.enable_attention_slicing()
image = pipeline(prompt).images[0]

id = str(int(time.time()))
image.save("output/universe/sd_moon_{}.png".format(id))
plt.imshow(image)
plt.axis('off')  # No axes for this image
plt.show()
