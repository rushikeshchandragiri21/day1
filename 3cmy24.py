import cv2
import numpy as np
from google.colab import files
import matplotlib.pyplot as plt

def rgb_to_cmy(image):
    # Normalize the RGB values
    normalized_image = image.astype(np.float32) / 255.0

    # Calculate the CMY values
    c = 1.0 - normalized_image[:, :, 0]  # Cyan
    m = 1.0 - normalized_image[:, :, 1]  # Magenta
    y = 1.0 - normalized_image[:, :, 2]  # Yellow

    # Stack the CMY channels
    cmy_image = np.stack((c, m, y), axis=-1) * 255.0

    # Convert to 8-bit unsigned integers
    cmy_image = cmy_image.astype(np.uint8)

    return cmy_image

# Upload the input image file
uploaded = files.upload()

# Read the uploaded image
image_path = list(uploaded.keys())[0]
input_image = cv2.imread(image_path)

# Display the input image
plt.imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('Input Image')
plt.show()

# Convert RGB image to CMY
cmy_image = rgb_to_cmy(input_image)

# Display the CMY image
plt.imshow(cmy_image)
plt.axis('off')
plt.title('CMY Image')
plt.show()

# Save the CMY image
cv2.imwrite('cmy_image.jpg', cmy_image)
