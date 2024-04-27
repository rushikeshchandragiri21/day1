import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files

def upload_image():
    uploaded = files.upload()
    for filename in uploaded.keys():
        img = cv2.imread(filename)
        return img

def compress_image(img, quality_factor):
    # Encode image using JPEG compression
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor]
    _, encoded_img = cv2.imencode('.jpg', img, encode_param)

    # Decode compressed image
    decoded_img = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)

    return decoded_img

def calculate_psnr(img1, img2):
    # Calculate Mean Squared Error (MSE)
    mse = np.mean((img1 - img2) ** 2)

    # Calculate Peak Signal-to-Noise Ratio (PSNR)
    max_pixel_value = 255
    psnr = 10 * np.log10((max_pixel_value ** 2) / mse)

    return psnr

def main():
    # Upload the original image
    original_img = upload_image()

    # Compress the original image using JPEG compression with quality factor 50
    compressed_img = compress_image(original_img, 50)

    # Calculate PSNR value comparing original and decompressed images
    psnr_original = calculate_psnr(original_img, original_img)
    psnr_compressed = calculate_psnr(original_img, compressed_img)

    # Plot the images
    plt.figure(figsize=(15, 6))

    # Plot original image
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    # Plot compressed image
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(compressed_img, cv2.COLOR_BGR2RGB))
    plt.title('Compressed Image (Quality Factor: 50)')
    plt.axis('off')

    plt.show()

    # Print PSNR values
    print(f'PSNR value for original image: {psnr_original:.2f} dB')
    print(f'PSNR value for compressed image: {psnr_compressed:.2f} dB')

if __name__ == "__main__":
    main()
