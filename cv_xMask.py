# multiply mask with image
import cv2
import numpy as np

# Create a white canvas of 1024x576
canvas = np.ones((576, 1024, 3), dtype=np.uint8) * 0
mask = cv2.imread("output/gradient.png")
original = cv2.imread("output/default.png")

# multiply mask with image
# convert mask to grayscale
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
# convert the mask to 0 to 1 then multiply
mask = mask.astype(float) / 255.0
# mask the image
for c in range(0, 3):
    original[:, :, c] = original[:, :, c] * mask
# display
cv2.imshow("masked", original)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the image
cv2.imwrite('output/masked.png', original)

