import numpy as np
import cv2
import matplotlib.pyplot as plt
from google.colab import files

def upload_image():
    uploaded = files.upload()
    for filename in uploaded.keys():
        img = cv2.imread(filename)
        return img

def calculate_histogram(img):
    # Calculate histogram for each channel
    hist = [np.histogram(img[:,:,i], bins=256, range=(0,256))[0] for i in range(3)]
    return hist

def calculate_cdf(hist):
    # Compute cumulative distribution function (CDF)
    cdf = [np.cumsum(hist[i]) for i in range(3)]
    return cdf

def normalize_cdf(cdf, img_size):
    # Normalize CDF to desired range
    normalized_cdf = [((cdf[i] - cdf[i].min()) / (img_size - 1) * 255).astype(np.uint8) for i in range(3)]
    return normalized_cdf

def map_intensity_values(img, normalized_cdf):
    # Map intensity values of the original image to new intensity values
    equalized_img = np.empty_like(img)
    for i in range(3):
        equalized_img[:,:,i] = np.interp(img[:,:,i], np.arange(256), normalized_cdf[i])
    return equalized_img

def histogram_equalization(img):
    # Calculate histogram
    hist = calculate_histogram(img)

    # Calculate CDF
    cdf = calculate_cdf(hist)

    # Normalize CDF
    normalized_cdf = normalize_cdf(cdf, img.size)

    # Map intensity values
    equalized_img = map_intensity_values(img, normalized_cdf)

    return equalized_img, hist, [np.histogram(normalized_cdf[i], bins=256, range=(0,256))[0] for i in range(3)]

def plot_histograms(hist_original, hist_equalized):
    # Plot original and equalized histograms
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(hist_original[0], color='red', label='Red Channel')
    plt.plot(hist_original[1], color='green', label='Green Channel')
    plt.plot(hist_original[2], color='blue', label='Blue Channel')
    plt.title('Original Image Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(hist_equalized[0], color='red', label='Red Channel')
    plt.plot(hist_equalized[1], color='green', label='Green Channel')
    plt.plot(hist_equalized[2], color='blue', label='Blue Channel')
    plt.title('Equalized Image Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

def main():
    # Upload the image
    img = upload_image()

    # Perform histogram equalization
    equalized_img, hist_original, hist_equalized = histogram_equalization(img)

    # Plot histograms
    plot_histograms(hist_original, hist_equalized)

    # Display original and equalized images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(equalized_img, cv2.COLOR_BGR2RGB))
    plt.title('Equalized Image')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()
