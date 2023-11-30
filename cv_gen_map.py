import cv2
import numpy as np

# Create a white canvas of 1024x576
canvas = np.ones((576, 1024, 3), dtype=np.uint8) * 0

# Circle parameters
center_circle_l = (76, 500)
center_circle_r = (948, 500)
radius = 72
for r in range(radius, 0, -1):
    # Gradient from black (outer) to white (center)
    color = int(255 * (1 - r / radius))
    cv2.circle(canvas, center_circle_l, r, (color, color, color), -1)
    cv2.circle(canvas, center_circle_r, r, (color, color, color), -1)



# Rectangle coordinates
top_left = (4, 0)
bottom_right = (76, 500)

# Drawing the gradient rectangle
for i in range(top_left[0], bottom_right[0]):
    color = int(255 * (i - top_left[0]) / (bottom_right[0] - top_left[0]))
    cv2.line(canvas, (i, top_left[1]), (i, bottom_right[1]), (color, color, color), 1)

top_left = (948, 0)
bottom_right = (1020, 500)
# Drawing the gradient rectangle
for i in range(top_left[0], bottom_right[0]):
    color = int(255 * (1 - (i - top_left[0]) / (bottom_right[0] - top_left[0])))
    cv2.line(canvas, (i, top_left[1]), (i, bottom_right[1]), (color, color, color), 1)

top_left = (76, 500)
bottom_right = (948, 572)
# Drawing the gradient from black (bottom) to white (top)
for i in range(top_left[1], bottom_right[1]):
    color = int(255 * (1 - (i - top_left[1]) / (bottom_right[1] - top_left[1])))
    cv2.line(canvas, (top_left[0], i), (bottom_right[0], i), (color, color, color), 1)

top_left = (76, 0)
bottom_right = (948, 500)
# draw a white rectangle to cover the center
cv2.rectangle(canvas, top_left, bottom_right, (255, 255, 255), -1)


# Display the image
cv2.imshow('gradient_rectangle', canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the image
cv2.imwrite('output/gradient.png', canvas)
