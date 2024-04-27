import cv2
import numpy as np
from google.colab import files
import matplotlib.pyplot as plt

def edge_detection_prewitt(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Define Prewitt kernels
    prewitt_x = np.array([[-1, -1, -1],
                          [ 0,  0,  0],
                          [ 1,  1,  1]])

    prewitt_y = np.array([[-1, 0, 1],
                          [-1, 0, 1],
                          [-1, 0, 1]])

    # Compute gradients using Prewitt operator
    grad_x = cv2.filter2D(blurred, -1, prewitt_x)
    grad_y = cv2.filter2D(blurred, -1, prewitt_y)

    # Calculate gradient magnitude
    magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # Threshold the magnitude to get binary edges
    threshold = 10
    edges = np.uint8(magnitude > threshold) * 255

    return edges

# Upload the input image file
uploaded = files.upload()

# Read the uploaded image
image_path = list(uploaded.keys())[0]
input_image = cv2.imread(image_path)

# Perform edge detection using Prewitt operator
edges = edge_detection_prewitt(input_image)

# Display the original and segmented images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(edges, cmap='gray')
plt.title('Segmented Image (Edges)')
plt.axis('off')

plt.show()
