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

def power_law(img, lam, plot = False, name = 'Null'):
    D = img.dtype.itemsize * 8
    c = pow((pow(2, D) - 1), (1-lam))
    result = c*np.power(img, lam)
    result = result.astype(img.dtype)
    if plot:
        fig = plt.figure("Spine Power Law Result with various lambda values")
        plotter(img_stretched, name)
    return result

img = cv.imread('salt-and-pepper-Skeleton.png', cv.IMREAD_GRAYSCALE)
D = img.dtype.itemsize * 8
print("Dimensions of the original chest image:", img.shape)
bins = np.arange(0, pow(2, D) + 1, 256) - 0.5
fig = plt.figure("Skeleton")
ax = plt.subplot(2, 2, 1)
plotter(img, "Original")
ax = plt.subplot(2, 2, 3)
plt.hist(img.ravel())


# P1: Removing Salt and Pepper noise
ksize = 3
img_noise_reduced = cv.medianBlur(img, ksize)
ax = plt.subplot(2, 2, 2)
plotter(img_noise_reduced, f"Noise Reduced-ksize={ksize}")
ax = plt.subplot(2, 2, 4)
plt.hist(img_noise_reduced.ravel())

# P2: Vertical Bones
gx = cv.Sobel(img_noise_reduced, 3, dx = 1, dy = 0)
gx = cv.convertScaleAbs(gx)
gx_smoothed = cv.filter2D(gx, -1, np.ones((5, 5))/25)
# fig = plt.figure("gx hist")
# plt.hist(gx.ravel())
fig = plt.figure("Vertical")
ax = plt.subplot(1, 3, 1)
plotter(img_noise_reduced, "Original")
ax = plt.subplot(1, 3, 2)
plotter(np.abs(gx_smoothed), 'X Gradient')
ax = plt.subplot(1, 3, 3)
plotter(img_noise_reduced+np.abs(gx_smoothed), 'Original + Gradient')

# P3: Horizontal Bones
gy = cv.Sobel(img_noise_reduced, 3, dx = 0, dy = 1)
gy = cv.convertScaleAbs(gy)
gy_smoothed = cv.filter2D(gy, -1, np.ones((5, 5))/25)
# fig = plt.figure("gy hist")
# plt.hist(gy.ravel())
fig = plt.figure("Horizontal")
ax = plt.subplot(1, 3, 1)
plotter(img_noise_reduced, "Original")
ax = plt.subplot(1, 3, 2)
plotter(np.abs(gy_smoothed), 'Y Gradient')
ax = plt.subplot(1, 3, 3)
plotter(img_noise_reduced+np.abs(gy_smoothed), 'Original + Gradient')

# P4: Seeing Muscle Tissue
mag = np.abs(gx) + np.abs(gy)
lap = cv.Laplacian(img_noise_reduced, 3, 3)
lap = cv.convertScaleAbs(lap)
sharpened = img_noise_reduced + lap
# print(cv.CV_16S)
# lap2 = laplacian(img_noise_reduced, 45)

smoothed_grad = cv.filter2D(mag, -1, np.ones((5, 5))/25)

fig = plt.figure("First Page")
ax = plt.subplot(1, 4, 1)
plotter(img_noise_reduced, '(a) Noise reduced original')
ax = plt.subplot(1, 4, 2)
plotter(lap, '(b) Laplacian')
ax = plt.subplot(1, 4, 3)
plotter(sharpened, '(c) Sharpened with CV Laplacian')
ax = plt.subplot(1, 4, 4)
plotter(mag, '(d) Sobel Magnitude of image')

fig = plt.figure("Second Page")
ax = plt.subplot(1, 4, 1)
plotter(smoothed_grad, '(e) Smoothed Grad')
ax = plt.subplot(1, 4, 2)
plotter(smoothed_grad*sharpened, '(f):(e) times (c)')
ax = plt.subplot(1, 4, 3)
plotter(img_noise_reduced + smoothed_grad*sharpened, '(g):(a) plus (f)')
ax = plt.subplot(1, 4, 4)
plotter(power_law(sharpened, 0.5), '(h):Power Law of (c)')

fig = plt.figure("Final Tissue")
plotter(power_law(sharpened+mag, 0.75), 'Final')

plt.show()