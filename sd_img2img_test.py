import torch
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline
# from diffusers import DiffusionPipeline
import matplotlib.pyplot as plt
from PIL import Image
import time

model_path = 'bl3uprint.safetensors'


checkpoint_path = "./stable-diffusion-v1-5"
# pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(checkpoint_path)
pipeline = StableDiffusionPipeline.from_pretrained(checkpoint_path, torch_dtype=torch.float16, use_safetensors=True)
# pipeline.enable_attention_slicing()
pipeline = pipeline.to("cuda")

# pipeline.load_lora_weights('./models', weight_name=model_path)

pipeline.load_lora_weights('./models', weight_name="400shijing.safetensors")
# pipeline.fuse_lora(lora_scale=0.7)
# pipeline.unet.load_attn_procs("./models", weight_name="400shijing.safetensors", local_files_only=True)
# pipeline.unet.load_attn_procs("jbilcke-hf/sdxl-cinematic-1", weight_name="pytorch_lora_weights.safetensors")


init_image = Image.open("output/weapon/t1_4_1699765031.png").convert("RGB")
init_image = init_image.resize((512, 512))


prompt = "shijing, a high end building with a lot of glass windows"

images = pipeline(prompt).images
# images = pipeline(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5).images

# pipe.enable_attention_slicing()
image = images[0]

id = str(int(time.time()))
image.save("output/shijing/sd_shijing_{}.png".format(id))

plt.imshow(image)
plt.axis('off')  # No axes for this image
plt.show()