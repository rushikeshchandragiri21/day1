import cv2
import numpy as np
import matplotlib.pyplot as plt

def intensity_slicing_preserve_background(image, lower_threshold, upper_threshold):
    # Create a copy of the original image
    sliced_image = np.copy(image)

    # Apply intensity slicing, preserving the background
    sliced_image[(image >= lower_threshold) & (image <= upper_threshold)] = 255

    return sliced_image

def intensity_slicing_non_preserve_background(image, lower_threshold, upper_threshold):
    # Create a copy of the original image
    sliced_image = np.copy(image)

    # Apply intensity slicing, non-preserving the background
    sliced_image[(image >= lower_threshold) & (image <= upper_threshold)] = 0

    return sliced_image

# Read the input image
image = cv2.imread('24bitcoloeimage.jpeg', cv2.IMREAD_GRAYSCALE)

# Define the threshold values
lower_threshold = 100
upper_threshold = 200

# Apply intensity slicing preserving the background
sliced_image_preserve_bg = intensity_slicing_preserve_background(image, lower_threshold, upper_threshold)

# Apply intensity slicing without preserving the background
sliced_image_non_preserve_bg = intensity_slicing_non_preserve_background(image, lower_threshold, upper_threshold)

# Plot the images
plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(sliced_image_preserve_bg, cmap='gray')
plt.title('With Preserve Background')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(sliced_image_non_preserve_bg, cmap='gray')
plt.title('Without Preserve Background')
plt.axis('off')

plt.show()
