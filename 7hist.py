import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files

# Function to plot histograms for individual RGB channels
def plot_histograms(image):
    # Split the image into its RGB channels
    b, g, r = cv2.split(image)

    # Compute histograms for each channel
    hist_b = cv2.calcHist([b], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([g], [0], None, [256], [0, 256])
    hist_r = cv2.calcHist([r], [0], None, [256], [0, 256])

    # Plot histograms
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 3, 1)
    plt.plot(hist_b, color='blue')
    plt.title('Blue Channel Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')

    plt.subplot(1, 3, 2)
    plt.plot(hist_g, color='green')
    plt.title('Green Channel Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')

    plt.subplot(1, 3, 3)
    plt.plot(hist_r, color='red')
    plt.title('Red Channel Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

# Upload an image file
uploaded = files.upload()

try:
    # Get the file name
    file_name = next(iter(uploaded))

    # Load the image
    image = cv2.imread(file_name)

    # Check if image is loaded successfully
    if image is not None:
        plot_histograms(image)
    else:
        print("Failed to load the image.")
except Exception as e:
    print("An error occurred:", e)
