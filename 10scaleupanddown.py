import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files

def upload_image():
    uploaded = files.upload()
    for filename in uploaded.keys():
        img = cv2.imread(filename)
        return img

def translate_image(img, dx, dy):
    # Define translation matrix
    M = np.float32([[1, 0, dx], [0, 1, dy]])

    # Apply translation
    translated_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

    return translated_img

def main():
    # Upload the image
    img = upload_image()

    # Translate the image to the right by 20 units
    translated_img_right = translate_image(img, 20, 0)

    # Translate the image downwards by 10 units
    translated_img_down = translate_image(img, 0, 10)

    # Plot the images
    plt.figure(figsize=(15, 10))

    # Plot original image and translation to the right
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(cv2.cvtColor(translated_img_right, cv2.COLOR_BGR2RGB))
    plt.title('Translated to Right by 20 units')
    plt.axis('off')

    # Plot original image and translation downwards
    plt.subplot(2, 2, 3)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(cv2.cvtColor(translated_img_down, cv2.COLOR_BGR2RGB))
    plt.title('Translated Downwards by 10 units')
    plt.axis('off')

    plt.show()

if __name__ == "__main__":
    main()
