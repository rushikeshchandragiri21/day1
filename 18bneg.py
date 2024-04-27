import cv2
from google.colab import files
import numpy as np
from IPython.display import display, Image

def negative_image(image):
    # Invert the pixel values
    negative = 255 - image
    return negative

# Upload the input image file
uploaded = files.upload()

# Read the uploaded image
image_path = list(uploaded.keys())[0]
input_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Check if the image is 8-bit
if input_image.dtype != np.uint8:
    print("Input image is not 8-bit grayscale.")
else:
    print("Input image is 8-bit grayscale.")
# Get the negative of the image
negative_result = negative_image(input_image)

# Save the result
cv2.imwrite('negative_image.jpg', negative_result)

# Display the input and negative images
display(Image(image_path))
display(Image('negative_image.jpg'))
