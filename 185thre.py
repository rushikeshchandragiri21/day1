import cv2
import numpy as np
from google.colab import files
from matplotlib import pyplot as plt

def apply_threshold(image_gray, threshold_value):
    # Apply thresholding
    _, binary_image = cv2.threshold(image_gray, threshold_value, 255, cv2.THRESH_BINARY)

    return binary_image

def main():
    # Upload the grayscale image
    uploaded = files.upload()

    # Read the uploaded image
    for filename in uploaded.keys():
        image_gray = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    # Specify the threshold value
    threshold_value = int(input("Enter the threshold value: "))

    # Apply thresholding
    binary_image = apply_threshold(image_gray, threshold_value)

    # Display the original and binary images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1), plt.imshow(image_gray, cmap='gray')
    plt.title('Original Image'), plt.axis('off')
    plt.subplot(1, 2, 2), plt.imshow(binary_image, cmap='binary')
    plt.title('Binary Image (Threshold={})'.format(threshold_value)), plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()
