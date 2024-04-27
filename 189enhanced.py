import cv2
import numpy as np
from google.colab import files
import matplotlib.pyplot as plt

def read_and_display_image(upload_function):
    uploaded_image = upload_function()
    uploaded_image_data = list(uploaded_image.values())[0]
    image = cv2.imdecode(np.frombuffer(uploaded_image_data, np.uint8), cv2.IMREAD_COLOR)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    plt.show()
    return image

def enhance_color_channel(image, channel_index, enhancement_factor):
    enhanced_image = image.copy()
    enhanced_image[:, :, channel_index] = np.clip(image[:, :, channel_index] * enhancement_factor, 0, 255)
    return enhanced_image

# Upload the 24-bit RGB image
print("Upload the 24-bit RGB image:")
image = read_and_display_image(files.upload)

# Choose the color channel to enhance (0 for Red, 1 for Green, 2 for Blue)
channel_index = 1  # Enhancing the Green channel

# Define the enhancement factor
enhancement_factor = 1.5  # Increasing the intensity by 50%

# Enhance the selected color channel
enhanced_image = enhance_color_channel(image, channel_index, enhancement_factor)

# Display the enhanced image
plt.imshow(cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB))
plt.title('Enhanced Image (Green Channel Enhanced)')
plt.axis('off')
plt.show()
