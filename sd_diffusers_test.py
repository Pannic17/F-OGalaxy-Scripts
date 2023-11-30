from diffusers import DDPMPipeline
import matplotlib.pyplot as plt
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Now import other libraries

# Load the DDPM pipeline
ddpm = DDPMPipeline.from_pretrained("google/ddpm-cat-256", use_safetensors=True).to("cuda")

# Generate the image
image = ddpm(num_inference_steps=50).images[0]

# Display the image using matplotlib
plt.imshow(image)
plt.axis('off')  # No axes for this image
plt.show()
