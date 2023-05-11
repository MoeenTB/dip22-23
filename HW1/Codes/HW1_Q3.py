# Importing modules
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Defining the transformation functions
def M_rot(angle):
    return np.float32([[np.cos(angle), -np.sin(angle), 0],
                       [np.sin(angle), np.cos(angle), 0],
                       [0, 0, 1]])
def M_trans(tx, ty):
    return np.float32([[1, 0, tx],
                       [0, 1, ty],
                       [0, 0, 1]])
def M_scale(cx, cy):
    return np.float32([[cx, 0, 0],
                       [0, cy, 0],
                       [0, 0, 1]])
def M_shear(sv, sh):
    return np.float32([[1, sv, 0],
                       [sh, 1, 0],
                       [0, 0, 1]])

# P1: Reading the image
img = cv.imread('HW1_Q3.png', cv.IMREAD_GRAYSCALE)
# print("Dimensions of the original image:", img.shape)
rows, cols = img.shape

# P2: Extracting the main image in a relatively correct orientation and size
M1 = np.dot(M_rot(np.radians(20)), M_trans(-300, -750))
img_tfd1 = cv.warpPerspective(img, M1, (cols, rows))
img_tfd2 = cv.warpPerspective(img_tfd1, M_scale(1.6, 2.2), (cols, rows))
img_tfd3 = cv.warpPerspective(img_tfd2, M_trans(20, -60), (cols, rows))
img_cropped = img_tfd3[:920, :]

# P3: Halving the original image's dimensions and keeping the black parts
old_size = (rows, cols)
rows, cols = img_cropped.shape
img_halved = cv.warpPerspective(img_cropped, M_scale(0.5, 0.5), (cols, rows))

# P3: Centering the original main image for the padding
img_centered = cv.warpPerspective(img_halved, M_trans(cols/4, rows/4), (cols, rows))
img_padded = np.zeros((rows*2, cols*2))
img_padded[int(rows/2):int(rows/2)+img_centered.shape[0], int(cols/2):int(cols/2)+img_centered.shape[1]] = img_centered

# P5: Translating the padded image and resampling
img_padded_translated = cv.warpPerspective(img_padded, M_trans(-820, -680), (cols*2, rows*2))
img_resmpl_NN = cv.warpPerspective(img_padded_translated, M_scale(4, 4), (cols*2, rows*2), flags = cv.INTER_NEAREST)
img_resmpl_LN = cv.warpPerspective(img_padded_translated, M_scale(4, 4), (cols*2, rows*2), flags = cv.INTER_LINEAR)
img_resmpl_CB = cv.warpPerspective(img_padded_translated, M_scale(4, 4), (cols*2, rows*2), flags = cv.INTER_CUBIC)

# img_resmpl_NN = cv.resize(img_padded, dsize = None, fx = 4, fy = 4, interpolation = cv.INTER_NEAREST)
# img_resmpl_LN = cv.resize(img_padded, dsize = None, fx = 4, fy = 4, interpolation = cv.INTER_LINEAR)
# img_resmpl_CB = cv.resize(img_padded, dsize = None, fx = 4, fy = 4, interpolation = cv.INTER_CUBIC)

ax = plt.subplot(3, 4, 1)
plt.axis('off')
plt.imshow(img, cmap = 'gray', vmin = 0, vmax = 255)

ax = plt.subplot(3, 4, 2)
plt.axis('off')
plt.imshow(img_tfd1, cmap = 'gray', vmin = 0, vmax = 255)

ax = plt.subplot(3, 4, 3)
plt.axis('off')
plt.imshow(img_tfd2, cmap = 'gray', vmin = 0, vmax = 255)

ax = plt.subplot(3, 4, 4)
plt.axis('off')
plt.imshow(img_tfd3, cmap = 'gray', vmin = 0, vmax = 255)

ax = plt.subplot(3, 4, 5)
plt.axis('off')
plt.imshow(img_cropped, cmap = 'gray', vmin = 0, vmax = 255)

ax = plt.subplot(3, 4, 6)
plt.axis('off')
plt.imshow(img_halved, cmap = 'gray', vmin = 0, vmax = 255)

ax = plt.subplot(3, 4, 7)
plt.axis('off')
plt.imshow(img_centered, cmap = 'gray', vmin = 0, vmax = 255)

ax = plt.subplot(3, 4, 8)
plt.axis('off')
plt.imshow(img_padded, cmap = 'gray', vmin = 0, vmax = 255)

ax = plt.subplot(3, 4, 9)
plt.axis('off')
plt.imshow(img_padded_translated, cmap = 'gray', vmin = 0, vmax = 255)

ax = plt.subplot(3, 4, 10)
plt.axis('off')
plt.imshow(img_resmpl_NN, cmap = 'gray', vmin = 0, vmax = 255)

ax = plt.subplot(3, 4, 11)
plt.axis('off')
plt.imshow(img_resmpl_LN, cmap = 'gray', vmin = 0, vmax = 255)

ax = plt.subplot(3, 4, 12)
plt.axis('off')
plt.imshow(img_resmpl_CB, cmap = 'gray', vmin = 0, vmax = 255)

fig = plt.figure()
plt.suptitle("Different Interpolation methods")

ax = fig.add_subplot(1, 3, 1)
plt.title('Nearest Neighbors interpolation')
plt.axis('off')
plt.imshow(img_resmpl_NN, cmap = 'gray', vmin = 0, vmax = 255)

ax = fig.add_subplot(1, 3, 2)
plt.title('Linear interpolation')
plt.axis('off')
plt.imshow(img_resmpl_LN, cmap = 'gray', vmin = 0, vmax = 255)

ax = fig.add_subplot(1, 3, 3)
plt.title('Cubic interpolation')
plt.axis('off')
plt.imshow(img_resmpl_CB, cmap = 'gray', vmin = 0, vmax = 255)

plt.show()
