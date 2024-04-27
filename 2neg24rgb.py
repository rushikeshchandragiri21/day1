import cv2
from google.colab import files
import matplotlib.pyplot as plt

def negative_image(image):
    # Check if the image is loaded successfully
    if image is None:
        print("Error: Unable to load image.")
        return None

    # Split the image into its color channels
    b, g, r = cv2.split(image)

    # Invert each color channel
    negative_b = 255 - b
    negative_g = 255 - g
    negative_r = 255 - r

    # Merge the inverted channels
    negative = cv2.merge((negative_b, negative_g, negative_r))

    return negative

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

# Get the negative of the image
negative_result = negative_image(input_image)

# Check if negative image is computed
if negative_result is not None:
    # Display the negative image
    plt.imshow(cv2.cvtColor(negative_result, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('Negative Image')
    plt.show()

    # Save the negative image
    cv2.imwrite('negative_image.jpg', negative_result)
