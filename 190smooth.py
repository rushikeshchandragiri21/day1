import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the noisy image
noisy_image = cv2.imread('13.jpeg')

# Convert the image to float32
noisy_image = noisy_image.astype(np.float32) / 255.0

# Apply Gaussian smoothing
smooth_image = cv2.GaussianBlur(noisy_image, (5, 5), 0)

# Display the original and smoothed images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB))
plt.title('Noisy Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(smooth_image, cv2.COLOR_BGR2RGB))
plt.title('Smoothed Image')
plt.axis('off')

plt.show()
