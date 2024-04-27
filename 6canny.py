import cv2
import numpy as np
from google.colab import files
import matplotlib.pyplot as plt

def edge_detection_canny(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to smooth the image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Compute gradient using Sobel operator
    grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

    # Calculate gradient magnitude and direction
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    direction = np.arctan2(grad_y, grad_x)

    # Apply non-maximal suppression
    suppressed_magnitude = np.zeros_like(magnitude)
    for i in range(1, magnitude.shape[0] - 1):
        for j in range(1, magnitude.shape[1] - 1):
            angle = direction[i, j]
            # Horizontal gradient
            if (0 <= angle < np.pi / 4) or (7 * np.pi / 4 <= angle < 2 * np.pi):
                if (magnitude[i, j] >= magnitude[i, j - 1]) and (magnitude[i, j] >= magnitude[i, j + 1]):
                    suppressed_magnitude[i, j] = magnitude[i, j]
            # Diagonal gradient (45 degrees)
            elif (np.pi / 4 <= angle < 3 * np.pi / 4):
                if (magnitude[i, j] >= magnitude[i - 1, j - 1]) and (magnitude[i, j] >= magnitude[i + 1, j + 1]):
                    suppressed_magnitude[i, j] = magnitude[i, j]
            # Vertical gradient
            elif (3 * np.pi / 4 <= angle < 5 * np.pi / 4):
                if (magnitude[i, j] >= magnitude[i - 1, j]) and (magnitude[i, j] >= magnitude[i + 1, j]):
                    suppressed_magnitude[i, j] = magnitude[i, j]
            # Diagonal gradient (135 degrees)
            else:
                if (magnitude[i, j] >= magnitude[i - 1, j + 1]) and (magnitude[i, j] >= magnitude[i + 1, j - 1]):
                    suppressed_magnitude[i, j] = magnitude[i, j]

    # Apply thresholding
    edges = np.zeros_like(suppressed_magnitude)
    low_threshold = 30
    high_threshold = 100
    strong_edge = 255
    weak_edge = 50
    strong_i, strong_j = np.where(suppressed_magnitude >= high_threshold)
    weak_i, weak_j = np.where((suppressed_magnitude >= low_threshold) & (suppressed_magnitude < high_threshold))
    edges[strong_i, strong_j] = strong_edge
    edges[weak_i, weak_j] = weak_edge

    # Hysteresis thresholding
    for i in range(1, suppressed_magnitude.shape[0] - 1):
        for j in range(1, suppressed_magnitude.shape[1] - 1):
            if edges[i, j] == weak_edge:
                if (edges[i-1:i+2, j-1:j+2] == strong_edge).any():
                    edges[i, j] = strong_edge
                else:
                    edges[i, j] = 0

    return edges

# Upload the input image file
uploaded = files.upload()

# Read the uploaded image
image_path = list(uploaded.keys())[0]
input_image = cv2.imread(image_path)

# Perform edge detection using Canny operator
edges = edge_detection_canny(input_image)

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
