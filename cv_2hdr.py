import time

import cv2
import numpy as np
from matplotlib import pyplot as plt


def png2hdr(image_path: str, output_path: str, default_path=True, alpha=False):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if alpha:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # image = cv2.convertScaleAbs(image, alpha=1, beta=-20)
        image = cv2.convertScaleAbs(image, alpha=2)
        image = cv2.subtract(image, (200,200,200,0))
        image = cv2.convertScaleAbs(image, alpha=0.8)
        # image = cv2.canny(image, 100, 200)
        plt.imshow(image)
        plt.axis('off')  # No axes for this image
        plt.show()
        path = "output/hdr/a{}.hdr"
    else:
        path = "output/hdr/{}.hdr"

    image_float = image.astype(np.float32)
    image_float /= 255.0
    image_float = image_float ** 2.2
    # Save the image in HDR format using imageio
    # imageio.imwrite('output_image.hdr', image_float, format='HDR-FI')
    if default_path:
        cv2.imwrite(path.format(int(time.time())), image_float)
    else:
        cv2.imwrite(output_path, image_float)

png2hdr("output/upscale/sd_inpaint_1701210575.png", "output/test1.hdr", default_path=True, alpha=True)

# def png2hdrA(image_path: str, )