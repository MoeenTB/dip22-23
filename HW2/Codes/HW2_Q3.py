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

# P1: Reading the image
img_chest = cv.imread('HW2_Q3_chest.tif', cv.IMREAD_ANYDEPTH)
fig = plt.figure("Chest Image")
plotter(img_chest, "Chest")
D = img_chest.dtype.itemsize * 8

# P2: Data dimensions and dtype
print("Dimensions of the original chest image:", img_chest.shape)
print("Data type of the original chest image:", img_chest.dtype, "which is", img_chest.dtype.itemsize * 8, "bits")

# P3: function defintion
def contrast_stretching(img, plot = False, name = 'Null'):
    D = img.dtype.itemsize * 8
    img_min = np.min(img)
    img_max = np.max(img)
    # img_stretched = np.zeros(img.shape)
    # for i in range(img.shape[0]):
    #     for j in range(img.shape[1]):
    #         img_stretched[i, j] = (img[i, j] - img_min) * ((pow(2, D) - 1) / (img_max - img_min))
    img_stretched = (img - img_min) * ((pow(2, D) - 1) / (img_max - img_min))
    img_stretched = img_stretched.astype(img.dtype)
    if plot:
        fig = plt.figure(name)
        plotter(img_stretched, name)
    return img_stretched

img_chest_stretched = contrast_stretching(img_chest)

# P4: Histograms
bins = np.arange(0, pow(2, D) + 1, 256) - 0.5 # 256 bin width results in 256 bins. Above this takes a lot of computation time
fig = plt.figure("Original and Stretched Chest Image and Histograms")
ax = fig.add_subplot(2, 2, 1)
plotter(img_chest, "Chest")
ax = fig.add_subplot(2, 2, 2)
plt.title('Histogram of the original image', color = 'white', backgroundcolor = 'green')
plt.hist(img_chest.ravel(), bins = bins, rwidth = 0.6)
ax = fig.add_subplot(2, 2, 3)
plotter(img_chest_stretched, "Stretched Chest")
ax = fig.add_subplot(2, 2, 4)
plt.title('Histogram of the stretched image', color = 'white', backgroundcolor = 'green')
plt.hist(img_chest_stretched.ravel(), bins = bins, rwidth = 0.6)


# P5: Reading the image
img_spine = cv.imread('HW2_Q3_spine.tif', cv.IMREAD_ANYDEPTH)
fig = plt.figure("Spine Image")
plotter(img_spine, "Spine")
D = img_spine.dtype.itemsize * 8
print("Dimensions of the original spine image:", img_spine.shape)
print("Data type of the original spine image:", img_spine.dtype, "which is", img_spine.dtype.itemsize * 8, "bits")

# P6:
def power_law(img, lam, plot = False, name = 'Null'):
    D = img.dtype.itemsize * 8
    c = pow((pow(2, D) - 1), (1-lam))
    result = c*np.power(img, lam)
    result = result.astype(img.dtype)
    if plot:
        fig = plt.figure("Spine Power Law Result with various lambda values")
        plotter(img_stretched, name)
    return result

# P7:
num = 12
lambd = np.linspace(1, 2, num)
fig = plt.figure("Spine after power law transformation with suitable lambda")
plt.suptitle('Spine PL')
for i, lmd in enumerate(lambd, 1):
    ax = fig.add_subplot(3, 4, i)
    img_pl = power_law(img_spine, lmd)
    plotter(img_pl, f"lambda={lmd:.2f}")
lamb_star = 1.55

# P8:
fig = plt.figure("Spine Image Transform Results")
plt.suptitle('Spine Image Transforms')
img_spine_stretched = contrast_stretching(img_spine)
img_spine_pl = power_law(img_spine, lamb_star)
ax = plt.subplot(1, 3, 1)
plotter(img_spine_stretched, "Stretched Spine Image")
ax = plt.subplot(1, 3, 2)
plotter(img_spine, "Original Spine Image")
ax = plt.subplot(1, 3, 3)
plotter(img_spine_pl, "Power Law Spine Image")

plt.show()