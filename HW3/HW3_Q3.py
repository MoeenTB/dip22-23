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
def contrast_stretching(img, plot = False, name = 'Null'):
    D = img.dtype.itemsize * 8
    img_min = np.min(img)
    img_max = np.max(img)
    img_stretched = (img - img_min) * ((pow(2, D) - 1) / (img_max - img_min))
    # img_stretched = img_stretched.astype(img.dtype)
    if plot:
        fig = plt.figure(name)
        plotter(img_stretched, name)
    return img_stretched

# P1: mean_median function definition
def mean_median(name, size, img):
    if name == 'mean':
        kernel = np.ones((size, size), np.float32) / (size * size)
        img_filtered = cv.filter2D(img, -1, kernel) # Convolvs img with the kernel and -1 means the same depth as the input image
    elif name == 'median':
        img_filtered = cv.medianBlur(img, size) # This does the median blurring and size is the kernel size
    else:
        print("Invalid name. Please enter 'mean' or 'median'.")
    return img_filtered

# Another version of the previous function, implemented manually
def mean_median_manual(name, size, img):
    m = img.shape[0]
    n = img.shape[1]
    img_padded = np.zeros((m + size - 1, n + size - 1), dtype = img.dtype) # Zero padding
    img_padded[int(size/2):int(size/2)+m, int(size/2):int(size/2)+n] = img
    img_filtered = np.zeros_like(img)
    if name == 'mean':
        kernel = np.ones((size, size), np.float32) / (size * size) # The normalized averaging kernel
        for i in range(m):
            for j in range(n):
                img_filtered[i, j] = np.sum(img_padded[i:i+size, j:j+size] * kernel)
    elif name == 'median':
        for i in range(m):
            for j in range(n):
                img_filtered[i, j] = np.median(img_padded[i:i+size, j:j+size])
    else:
        print("Invalid name. Please enter 'mean' or 'median'.")
    return img_filtered

# P3: The laplacian function (second derivative)
def laplacian(img, n):
    if n == 45:
        kernel = np.ones((3, 3))
        kernel[1, 1] = -8
    elif n == 90:
        kernel = np.array([ [0, 1, 0],
                            [1, -4, 1],
                            [0, 1, 0]])
    else:
        print("Invalid n. Please enter 45 or 90.")
    img_filtered = cv.filter2D(img, -1, kernel)
    return img_filtered

# Reading the image
img = cv.imread('Kidney.png', cv.IMREAD_GRAYSCALE)
print("Dimensions of the original chest image:", img.shape)
D = img.dtype.itemsize * 8

# P2 : Performing mean and median on the image
img_mean = mean_median('mean', 5, img)
img_median = mean_median('median', 5, img)

fig = plt.figure("Kidney Original and Filtered")
ax = plt.subplot(1, 3, 1)
plotter(img, "Kidney")
ax = plt.subplot(1, 3, 2)
plotter(img_mean, "Kidney Mean Filtered")
ax = plt.subplot(1, 3, 3)
plotter(img_median, "Kidney Median Filtered")

# P4: Performing the laplacian function with n = 45 and n = 90 on the image
img_lap_45 = laplacian(img, 45)
img_lap_45 = cv.convertScaleAbs(img_lap_45) # To convert the image to 8-bit unsigned integer
# img_lap_45 = contrast_stretching(img_lap_45)
img_lap_45 = cv.createCLAHE(clipLimit = 10.0, tileGridSize = (8, 8)).apply(img_lap_45) # To obtain better contrast
img_lap_90 = laplacian(img, 90)
img_lap_90 = cv.convertScaleAbs(img_lap_90)
# img_lap_90 = contrast_stretching(img_lap_90)
img_lap_90 = cv.createCLAHE(clipLimit = 10.0, tileGridSize = (8, 8)).apply(img_lap_90)
print(np.min(img_lap_45), np.max(img_lap_45))
print(np.min(img_lap_90), np.max(img_lap_90))
fig = plt.figure()
plt.hist(img_lap_45.ravel())

fig = plt.figure("Kidney Lapclians")
ax = plt.subplot(1, 2, 1)
plotter(img_lap_45, "45 degree isotropic")
ax = plt.subplot(1, 2, 2)
plotter(img_lap_90, "90 degree isotropic")

plt.show()