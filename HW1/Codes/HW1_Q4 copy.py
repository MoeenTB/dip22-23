import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

bg = cv.imread('HW1_Q4_background_LE.bmp', cv.IMREAD_GRAYSCALE)
fs = cv.imread('HW1_Q4_fullscale_LE.bmp', cv.IMREAD_GRAYSCALE)
obj = cv.imread('HW1_Q4_object_LE.bmp', cv.IMREAD_GRAYSCALE)
print("Dimensions of the background LE:", bg.shape)
print("Dimensions of the fullscale LE:", fs.shape)
print("Dimensions of the object LE:", obj.shape)

x = np.array([[1, 2, 5],
              [3, 4, 6]])
a = np.array([1, 6])
a = np.reshape(a, (a.shape[0], 1))
print(x-a)
q = np.array([1, 2, 3])
q = np.reshape(q, (q.shape[0], 1))
print(q.shape)
print(q)
ww = np.repeat(q, 2, axis = 1)
print(ww)
print(np.mean(x, axis = 1))
bg_mean = np.mean(bg, axis = 1)
fs_mean = np.mean(fs, axis = 1)
m = 255/(fs_mean-bg_mean)
print(m.shape)
m = np.reshape(m, (m.shape[0], 1))
m = np.repeat(m, obj.shape[1], axis = 1)
print(m.shape)
bg_mean = np.reshape(bg_mean, (bg_mean.shape[0], 1))

obj_normalized = (obj - bg_mean)/m
print(obj_normalized.shape)

alpha = 5 # Simple contrast control
beta = 9    # Simple brightness control

obj_enhanced = cv.convertScaleAbs(obj_normalized, alpha=alpha, beta=beta)
# Instead you could do this element-by-element:
# new_image[y,x,c] = np.clip(alpha*image[y,x,c] + beta, 0, 255)

ax = plt.subplot(1, 3, 1)
plt.title('Raw Image')
plt.axis('off')
plt.imshow(obj, cmap = 'gray', vmin = 0, vmax = 255)

ax = plt.subplot(1, 3, 2)
plt.title('Normalized Image')
plt.axis('off')
plt.imshow(obj_normalized, cmap = 'gray', vmin = 0, vmax = 255)

ax = plt.subplot(1, 3, 3)
plt.title('Enhanced Image')
plt.axis('off')
plt.imshow(obj_enhanced, cmap = 'gray', vmin = 0, vmax = 255)

plt.show()

