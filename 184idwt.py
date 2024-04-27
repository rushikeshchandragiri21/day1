import cv2
import numpy as np
import pywt
from google.colab import files
from google.colab.patches import cv2_imshow

def apply_dwt_and_idwt(original_image_path):
    # Read the original image
    original_image = cv2.imread(original_image_path)

    # Convert the original image to grayscale
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Apply Discrete Wavelet Transform (DWT)
    coeffs = pywt.dwt2(gray_image, 'haar')

    # Get the coefficients
    cA, (cH, cV, cD) = coeffs

    # Inverse Discrete Wavelet Transform (IDWT)
    reconstructed_image = pywt.idwt2((cA, (cH, cV, cD)), 'haar')

    # Scale the reconstructed image to the same range as the original grayscale image
    reconstructed_image = np.clip(reconstructed_image, 0, 255)
    reconstructed_image = np.uint8(reconstructed_image)

    # Calculate PSNR (Peak Signal to Noise Ratio)
    psnr = cv2.PSNR(gray_image, reconstructed_image)

    # Display original and reconstructed images
    cv2_imshow(gray_image)
    cv2_imshow(reconstructed_image)
    print(f"PSNR Value: {psnr} dB")

def main():
    # Upload the original image
    uploaded = files.upload()

    # Read the uploaded image
    for filename in uploaded.keys():
        original_image_path = filename

    # Apply DWT and IDWT
    apply_dwt_and_idwt(original_image_path)

if __name__ == "__main__":
    main()
