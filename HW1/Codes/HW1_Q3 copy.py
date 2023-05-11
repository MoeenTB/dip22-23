import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('HW1_Q3.png')
print("Dimensions of the original image:", img.shape)
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
print("Dimensions of the grayscaled image:", img_gray.shape)

img2 = cv.imread('HW1_Q3.png', cv.IMREAD_GRAYSCALE)
print("Dimensions of the original image2:", img2.shape)
print("img dtype:", img.dtype)
print("img_gray dtype:", img_gray.dtype)
print("img2 dtype:", img2.dtype)
# print("img chnl0", np.unique(img[:,:,0]))
# print("img chnl1", np.unique(img[:,:,1]))
# print("img chnl2", np.unique(img[:,:,2]))
print(np.unique(img[:,:,1] - img[:,:, 2]))
ax = plt.subplot(1, 4, 1)
plt.imshow(img, cmap = 'gray', vmin = 0, vmax = 255)

ax = plt.subplot(1, 4, 2)
plt.imshow(img)

ax = plt.subplot(1, 4, 3)
plt.imshow(img_gray, cmap = 'gray')

ax = plt.subplot(1, 4, 4)
plt.imshow(img2, cmap = 'gray')

# cv.imshow('Test',img)
# cv.imshow('Test2',img_gray)
# cv.waitKey(0)
# cv.destroyAllWindows()

# m = np.array([[1, 0, 10],
#              [0, 1, 0],
#              [0, 0, 1]])
# x = cv.transform(img_gray, m)
# # cv.rotate(src, rotateCode)
# plt.imshow(x)
# plt.show()
# img_m = 

# # print(type(img_gray))
# # print("Type of each pixel in original image:", img.dtype)
# print("Type of each pixel in grayscaled image:", img_gray.dtype)

# img_cropped = img_gray[75:, 310:]
# img_cropped = img_cropped[:, :180]

# img_flipped = cv.flip(img_cropped, 0) # Which x axis? The x axis in DIP or the more conventional horizontal one?

# bins = np.arange(0, 257, 4) - 0.5

# # 1 1
# ax = plt.subplot(2, 3, 1)
# plt.suptitle("Versions of Image and their histograms")
# plt.title('Grayscale Image')
# plt.axis('off')
# plt.imshow(img_gray, cmap = 'gray', vmin=0, vmax=255)

# # 1 2
# ax = plt.subplot(2, 3, 2)
# plt.title('Cropped Image')
# plt.axis('off')
# plt.imshow(img_cropped, cmap = 'gray', vmin=0, vmax=255)

# # 1 3
# ax = plt.subplot(2, 3, 3)
# plt.title('Flipped Image')
# plt.axis('off')
# plt.imshow(img_flipped, cmap = 'gray', vmin=0, vmax=255)

# # 2 1
# ax = plt.subplot(2, 3, 4)
# plt.title('Grayscale Image Histogram')
# plt.hist(img_gray.ravel(), bins = bins, rwidth = 0.6)

# # 2 2
# ax = plt.subplot(2, 3, 5)
# plt.title('Cropped Image Histogram')
# plt.hist(img_cropped.ravel(), bins = bins, rwidth = 0.6)

# # 2 3
# ax = plt.subplot(2, 3, 6)
# plt.title('Flipped Image Histogram')
# plt.hist(img_flipped.ravel(), bins = bins, rwidth = 0.6)

# plt.show()