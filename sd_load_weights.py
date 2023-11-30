import matplotlib.pyplot as plt
from diffusers import AutoPipelineForText2Image, StableDiffusionControlNetPipeline
import torch

pipeline = AutoPipelineForText2Image.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16).to("cuda")
# pipeline.load_lora_weights("./models", weight_name="pytorch_lora_weights.safetensors")
pipeline.unet.load_attn_procs("./models", weight_name="pytorch_lora_weights.safetensors")
# pipeline.unet.load_attn_procs("jbilcke-hf/sdxl-cinematic-1", weight_name="pytorch_lora_weights.safetensors")

# use cnmt in the prompt to trigger the LoRA



prompt = "A cute cnmt eating a slice of pizza, stunning color scheme, masterpiece, illustration"
image = pipeline(prompt).images[0]


plt.imshow(image)
plt.axis('off')  # No axes for this image
plt.show()