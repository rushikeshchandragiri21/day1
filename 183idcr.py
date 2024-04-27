import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('13.jpeg')

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply DCT (Discrete Cosine Transform)
dct_image = cv2.dct(np.float32(gray_image))

# Normalize DCT coefficients for display (optional)
dct_image_norm = cv2.normalize(dct_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# Inverse DCT to retrieve the image
idct_image = cv2.idct(dct_image)

# Convert idct_image to the same data type as gray_image
idct_image = np.uint8(idct_image)

# Calculate PSNR between original and IDCT image
psnr_value = cv2.PSNR(gray_image, idct_image)

# Display the original and IDCT images
plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)
plt.imshow(gray_image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(idct_image, cmap='gray')
plt.title('IDCT Image')
plt.axis('off')

plt.tight_layout()
plt.show()

# Print the PSNR value
print("PSNR value:", psnr_value)
