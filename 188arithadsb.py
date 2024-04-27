import cv2
import matplotlib.pyplot as plt
import numpy as np
from google.colab import files

# Function to read and display images
def read_and_display_image(upload_function):
    # Upload the image
    uploaded_image = upload_function()
    # Get the uploaded image data
    uploaded_image_data = list(uploaded_image.values())[0]
    # Read the uploaded image
    image = cv2.imdecode(np.frombuffer(uploaded_image_data, np.uint8), cv2.IMREAD_GRAYSCALE)
    # Convert the image data type to float
    image = image.astype(float)
    # Display the image using Matplotlib
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()
    return image

# Function to resize images to the same dimensions
def resize_images(image1, image2):
    height = min(image1.shape[0], image2.shape[0])
    width = min(image1.shape[1], image2.shape[1])
    image1_resized = cv2.resize(image1, (width, height))
    image2_resized = cv2.resize(image2, (width, height))
    return image1_resized, image2_resized

# Function to perform addition of two images
def add_images(image1, image2):
    result = cv2.add(image1, image2)
    return result

# Function to perform subtraction of two images
def subtract_images(image1, image2):
    result = cv2.subtract(image1, image2)
    return result

# Upload the first image
print("Upload the first image:")
image1 = read_and_display_image(files.upload)

# Upload the second image
print("Upload the second image:")
image2 = read_and_display_image(files.upload)

# Resize images to the same dimensions
image1_resized, image2_resized = resize_images(image1, image2)

# Perform addition of two images
add_result = add_images(image1_resized, image2_resized)
plt.imshow(add_result, cmap='gray')
plt.title('Addition Result')
plt.axis('off')
plt.show()

# Perform subtraction of two images
subtract_result = subtract_images(image1_resized, image2_resized)
plt.imshow(subtract_result, cmap='gray')
plt.title('Subtraction Result')
plt.axis('off')
plt.show()
