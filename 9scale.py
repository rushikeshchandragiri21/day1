import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files

def upload_image():
    uploaded = files.upload()
    for filename in uploaded.keys():
        img = cv2.imread(filename)
        return img

def scale_image_nearest_neighbor(img, scale_factor):
    # Calculate new dimensions
    new_width = int(img.shape[1] * scale_factor)
    new_height = int(img.shape[0] * scale_factor)

    # Resize image using nearest neighbor interpolation
    scaled_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

    return scaled_img

def scale_image_bilinear(img, scale_factor):
    # Calculate new dimensions
    new_width = int(img.shape[1] * scale_factor)
    new_height = int(img.shape[0] * scale_factor)

    # Resize image using bilinear interpolation
    scaled_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    return scaled_img

def main():
    # Upload the image
    img = upload_image()

    # Scale the image by factor 2 using nearest neighbor interpolation
    scaled_img_nn = scale_image_nearest_neighbor(img, 2)

    # Scale the image by factor 0.5 using nearest neighbor interpolation
    scaled_img_nn_inv = scale_image_nearest_neighbor(img, 0.5)

    # Scale the image by factor 2 using bilinear interpolation
    scaled_img_bl = scale_image_bilinear(img, 2)

    # Scale the image by factor 0.5 using bilinear interpolation
    scaled_img_bl_inv = scale_image_bilinear(img, 0.5)

    # Plot the images
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(scaled_img_nn, cv2.COLOR_BGR2RGB))
    plt.title('Scaled by factor 2 (Nearest Neighbor Interpolation)')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(cv2.cvtColor(scaled_img_nn_inv, cv2.COLOR_BGR2RGB))
    plt.title('Scaled by factor 0.5 (Nearest Neighbor Interpolation)')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(cv2.cvtColor(scaled_img_bl, cv2.COLOR_BGR2RGB))
    plt.title('Scaled by factor 2 (Bilinear Interpolation)')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(cv2.cvtColor(scaled_img_bl_inv, cv2.COLOR_BGR2RGB))
    plt.title('Scaled by factor 0.5 (Bilinear Interpolation)')
    plt.axis('off')

    plt.show()

if __name__ == "__main__":
    main()
