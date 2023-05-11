# Importing modules
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def plotter(img, name):
    D = img.dtype.itemsize * 8
    plt.title(name, color = 'white', backgroundcolor = 'green')
    plt.axis('off')
    plt.imshow(img, cmap = 'gray', vmin = 0, vmax = pow(2, D) - 1)
    return

def plotter_hist_cdf(img, name, col):
    D = img.dtype.itemsize * 8
    img_hist, bins = np.histogram(img.ravel(), bins = np.arange(0, pow(2, D) + 1, 4) - 0.5, density = False)
    cdf = np.cumsum(img_hist)
    ax = fig.add_subplot(3, 3, col)
    plt.plot(img_hist, color = 'green')
    plt.title(name + " hist", color = 'white', backgroundcolor = 'green')
    ax = fig.add_subplot(3, 3, col+3)
    plt.plot(cdf/cdf[-1], color = 'red')  # It is divided by the final value of the cumulative sum to be normalized.
    plt.title(name + " CDF", color = 'white', backgroundcolor = 'green')
    return

# P1: Reading the image
# Fetus Image
img_fetus = cv.imread('HW2_Q4_Ultrasound-Fetus.tif', cv.IMREAD_GRAYSCALE)
fig = plt.figure("Original Fetus Image")
plotter(img_fetus, "Fetus")
D_fetus = img_fetus.dtype.itemsize * 8
print("Dimensions of the original chest image:", img_fetus.shape)
print("Data type of the original chest image:", img_fetus.dtype, "which is", img_fetus.dtype.itemsize * 8, "bits")

# Fingerprint Image
img_fingerprint = cv.imread('HW2_Q4_Fingerprint.tif', cv.IMREAD_GRAYSCALE)
fig = plt.figure("Original Fingerprint Image")
plotter(img_fingerprint, "Fingerprint")
D_fingerprint = img_fingerprint.dtype.itemsize * 8
print("Dimensions of the original fingerprint image:", img_fingerprint.shape)
print("Data type of the original fingerprint image:", img_fingerprint.dtype, "which is", img_fingerprint.dtype.itemsize * 8, "bits")

# P2: Custom HE function
def histogram_equalization(img):
    D = img.dtype.itemsize * 8
    img_hist, bins = np.histogram(img.ravel(), bins = np.arange(0, pow(2, D) + 1, 1) - 0.5, density = True)
    img_cdf = np.cumsum(img_hist)
    # First Method:
    # img_equalized = img_cdf[img] * (pow(2, D) - 1)
    # Second Method:
    cdf_m = np.ma.masked_equal(img_cdf,0)
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    img_equalized = cdf[img]
    
    img_equalized = img_equalized.astype(img.dtype)
    return img_equalized

# P3: perform_CLAHE function
def perform_CLAHE(img):
    result = cv.createCLAHE(clipLimit = 2.0, tileGridSize = (8, 8)).apply(img)
    return result

# P4: Performing HE on the images
custom_fetus = histogram_equalization(img_fetus)
custom_fingerprint = histogram_equalization(img_fingerprint)
CLAHE_fetus = perform_CLAHE(img_fetus)
CLAHE_fingerprint = perform_CLAHE(img_fingerprint)

bins = np.arange(0, pow(2, D_fetus) + 1, 4) - 0.5
# Plotting the fetus images and histograms
fig = plt.figure("Fetus Images and Histograms")
plt.suptitle("Fetus Images and Histograms", color = 'white', backgroundcolor = 'green')
ax = fig.add_subplot(3, 3, 1)
plotter(img_fetus, "Original Fetus")
ax = fig.add_subplot(3, 3, 2)
plotter(custom_fetus, "Custom HE Fetus")
ax = fig.add_subplot(3, 3, 3)
plotter(CLAHE_fetus, "CLAHE Fetus")
plotter_hist_cdf(img_fetus, "Original Fetus", 4)
plotter_hist_cdf(custom_fetus, "Custom HE Fetus", 5)
plotter_hist_cdf(CLAHE_fetus, "CLAHE Fetus", 6)

# Plotting the fingerprint images and histograms
fig = plt.figure("Fingerprint Images and Histograms")
plt.suptitle("Fingerprint Images and Histograms", color = 'white', backgroundcolor = 'green')
ax = fig.add_subplot(3, 3, 1)
plotter(img_fingerprint, "Original Fingerprint")
ax = fig.add_subplot(3, 3, 2)
plotter(custom_fingerprint, "Custom HE Fingerprint")
ax = fig.add_subplot(3, 3, 3)
plotter(CLAHE_fingerprint, "CLAHE Fingerprint")
plotter_hist_cdf(img_fingerprint, "Original Fingerprint", 4)
plotter_hist_cdf(custom_fingerprint, "Custom HE Fingerprint", 5)
plotter_hist_cdf(CLAHE_fingerprint, "CLAHE Fingerprint", 6)

plt.show()

