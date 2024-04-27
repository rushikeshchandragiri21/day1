import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files

def upload_image():
    uploaded = files.upload()
    for filename in uploaded.keys():
        img = cv2.imread(filename)
        return img

def rotate_image_clockwise(img):
    # Get image dimensions
    height, width = img.shape[:2]

    # Calculate rotation matrix
    M = cv2.getRotationMatrix2D((width / 2, height / 2), -90, 1)

    # Apply rotation
    rotated_img = cv2.warpAffine(img, M, (width, height))

    return rotated_img

def rotate_image_anticlockwise(img):
    # Get image dimensions
    height, width = img.shape[:2]

    # Calculate rotation matrix
    M = cv2.getRotationMatrix2D((width / 2, height / 2), 90, 1)

    # Apply rotation
    rotated_img = cv2.warpAffine(img, M, (width, height))

    return rotated_img

def main():
    # Upload the image
    img = upload_image()

    # Rotate the image clockwise by 90 degrees
    rotated_img_clockwise = rotate_image_clockwise(img)

    # Rotate the image anti-clockwise by 90 degrees
    rotated_img_anticlockwise = rotate_image_anticlockwise(img)

    # Plot the images
    plt.figure(figsize=(15, 10))

    # Plot original image and clockwise rotation
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(cv2.cvtColor(rotated_img_clockwise, cv2.COLOR_BGR2RGB))
    plt.title('Rotated Clockwise by 90°')
    plt.axis('off')

    # Plot original image and anti-clockwise rotation
    plt.subplot(2, 2, 3)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(cv2.cvtColor(rotated_img_anticlockwise, cv2.COLOR_BGR2RGB))
    plt.title('Rotated Anti-clockwise by 90°')
    plt.axis('off')

    plt.show()

if __name__ == "__main__":
    main()
