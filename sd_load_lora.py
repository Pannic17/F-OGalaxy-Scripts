from diffusers import StableDiffusionXLPipeline
import torch
import matplotlib.pyplot as plt


pipeline = StableDiffusionXLPipeline.from_pretrained(
    "Lykon/dreamshaper-xl-1-0", torch_dtype=torch.float16, variant="fp16"
).to("cuda")
pipeline.load_lora_weights("./models", weight_name="bl3uprint.safetensors")
# pipeline.unet.load_attn_procs("./models", weight_name="bl3uprint.safetensors", local_files_only=True)


prompt = "bl3uprint, a highly detailed blueprint of the empire state building, explaining how to build all parts, many txt, blueprint grid backdrop"
negative_prompt = "lowres, cropped, worst quality, low quality, normal quality, artifacts, signature, watermark, username, blurry, more than one bridge, bad architecture"

image = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    generator=torch.manual_seed(0),
).images[0]

pipeline.save_lora_weights("./models", weight_name="bl3uprint-R.safetensors")

plt.imshow(image)
plt.axis('off')  # No axes for this image
plt.show()