import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# P1: Reading the images
bg = cv.imread('HW1_Q4_background_LE.bmp', cv.IMREAD_GRAYSCALE)
fs = cv.imread('HW1_Q4_fullscale_LE.bmp', cv.IMREAD_GRAYSCALE)
obj = cv.imread('HW1_Q4_object_LE.bmp', cv.IMREAD_GRAYSCALE)
print("Dimensions of the background LE:", bg.shape)
print("Dimensions of the fullscale LE:", fs.shape)
print("Dimensions of the object LE:", obj.shape)

# P2: Calculating mean of columns
bg_mean = np.mean(bg, axis = 1, keepdims = True)
fs_mean = np.mean(fs, axis = 1, keepdims = True)

# P3: Mapping the range of each sensor to (0 to 255) and normalizing the image
m = 255/(fs_mean-bg_mean)
# m = np.reshape(m, (m.shape[0], 1))
print(m.shape)
m = np.repeat(m, obj.shape[1], axis = 1)
# bg_mean = np.reshape(bg_mean, (bg_mean.shape[0], 1))

obj_normalized = (obj - bg_mean)/m

# P4: Plotting the raw and normalized images side by side
ax = plt.subplot(1, 3, 1)
plt.title('Raw Image')
plt.axis('off')
plt.imshow(obj, cmap = 'gray', vmin = 0, vmax = 255)

ax = plt.subplot(1, 3, 2)
plt.title('Normalized Image')
plt.axis('off')
plt.imshow(obj_normalized, cmap = 'gray', vmin = 0, vmax = 255)

# P5: Implementing the contrast enhancement with the help of the open cv documentation

alpha = 5 # Simple contrast control
beta = 15    # Simple brightness control

obj_enhanced = cv.convertScaleAbs(obj_normalized, alpha=alpha, beta=beta)
# Instead, you could do this element-by-element:
# new_image[y,x,c] = np.clip(alpha*image[y,x,c] + beta, 0, 255)

ax = plt.subplot(1, 3, 3)
plt.title('Enhanced Image')
plt.axis('off')
plt.imshow(obj_enhanced, cmap = 'gray', vmin = 0, vmax = 255)

plt.show()

