import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files

def min_max_normalization(image):
    # Convert image to floating point representation
    normalized_image = image.astype(np.float32)

    # Normalize each channel separately
    for i in range(3):
        min_val = np.min(normalized_image[:,:,i])
        max_val = np.max(normalized_image[:,:,i])
        normalized_image[:,:,i] = (normalized_image[:,:,i] - min_val) / (max_val - min_val)

    # Scale pixel values back to [0, 255] range
    normalized_image = (normalized_image * 255).astype(np.uint8)

    return normalized_image

# Upload image file
uploaded = files.upload()

# Read the uploaded image
for name, data in uploaded.items():
    image = cv2.imdecode(np.frombuffer(data, np.uint8), -1)

# Perform Min-Max normalization
normalized_image = min_max_normalization(image)

# Display original and normalized images using Matplotlib
plt.figure(figsize=(10, 5))

# Original Image
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

# Normalized Image
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(normalized_image, cv2.COLOR_BGR2RGB))
plt.title('Normalized Image')
plt.axis('off')

plt.show()
