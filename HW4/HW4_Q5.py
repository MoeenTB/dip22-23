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
def plotterf(img, name):
    D = img.dtype.itemsize * 8
    plt.title(name, color = 'white', backgroundcolor = 'green')
    plt.axis('off')
    plt.imshow(img, cmap = 'gray')
    return
def M_rot(angle):
    return np.float32([[np.cos(angle), -np.sin(angle), 0],
                       [np.sin(angle), np.cos(angle), 0],
                       [0, 0, 1]])
def M_trans(tx, ty):
    return np.float32([[1, 0, tx],
                       [0, 1, ty],
                       [0, 0, 1]])
def contrast_stretching(img, plot = False, name = 'Null'):
    D = img.dtype.itemsize * 8
    img_min = np.min(img)
    img_max = np.max(img)
    img_stretched = (img - img_min) * ((pow(2, D) - 1) / (img_max - img_min))
    img_stretched = img_stretched.astype(img.dtype)
    if plot:
        fig = plt.figure(name)
        plotter(img_stretched, name)
    return img_stretched

# P1: Creating a blank image with a rectangle
img = np.zeros((200, 200), dtype = np.uint8)
rows, cols = img.shape
img[50:150, 80:120] = 255
plt.figure('P1')
plotter(img, 'Original Image')

# P2: Transformations of the image
img_transx = cv.warpPerspective(img, M_trans(30, 0), (cols, rows))
img_transy = cv.warpPerspective(img, M_trans(0, 20), (cols, rows))

img_rot1 = cv.warpAffine(img, cv.getRotationMatrix2D((cols / 2, rows / 2), 40, 1), (cols, rows))
img_rot2 = cv.warpAffine(img, cv.getRotationMatrix2D((cols / 2, rows / 2), -90, 1), (cols, rows))
# img_rot1 = cv.warpPerspective(img, cv(np.radians(-40)), (cols, rows))
# img_rot2 = cv.warpPerspective(img, M_rot(np.radians(90)), (cols, rows))

# P3: Taking DFT of the images
f_img = np.fft.fft2(img)
f_img_transx = np.fft.fft2(img_transx)
f_img_transy = np.fft.fft2(img_transy)
f_img_rot1 = np.fft.fft2(img_rot1)
f_img_rot2 = np.fft.fft2(img_rot2)

# Shifting the FFTs to the center
f_img = np.fft.fftshift(f_img)
f_img_transx = np.fft.fftshift(f_img_transx)
f_img_transy = np.fft.fftshift(f_img_transy)
f_img_rot1 = np.fft.fftshift(f_img_rot1)
f_img_rot2 = np.fft.fftshift(f_img_rot2)

# Plotting
plt.figure('P3')
ax = plt.subplot(5, 3, 1)
plotter(img, 'Original Image')
ax = plt.subplot(5, 3, 2)
plotterf(np.log(1 + np.abs(f_img)), 'DFT of Original Image')
ax = plt.subplot(5, 3, 3)
plotterf(np.angle(f_img), 'Phase of DFT of Original Image')
ax = plt.subplot(5, 3, 4)
plotter(img_transx, 'Translated Image in x')
ax = plt.subplot(5, 3, 5)
plotterf(np.log(1 + np.abs(f_img_transx)), 'DFT of Translated Image in x')
ax = plt.subplot(5, 3, 6)
plotterf(np.angle(f_img_transx), 'Phase of DFT of Translated Image in x')
ax = plt.subplot(5, 3, 7)
plotter(img_transy, 'Translated Image in y')
ax = plt.subplot(5, 3, 8)
plotterf(np.log(1 + np.abs(f_img_transy)), 'DFT of Translated Image in y')
ax = plt.subplot(5, 3, 9)
plotterf(np.angle(f_img_transy), 'Phase of DFT of Translated Image in y')
ax = plt.subplot(5, 3, 10)
plotter(img_rot1, 'Rotated Image 1')
ax = plt.subplot(5, 3, 11)
plotterf(np.log(1 + np.abs(f_img_rot1)), 'DFT of Rotated Image 1')
ax = plt.subplot(5, 3, 12)
plotterf(np.angle(f_img_rot1), 'Phase of DFT of Rotated Image 1')
ax = plt.subplot(5, 3, 13)
plotter(img_rot2, 'Rotated Image 2')
ax = plt.subplot(5, 3, 14)
plotterf(np.log(1 + np.abs(f_img_rot2)), 'DFT of Rotated Image 2')
ax = plt.subplot(5, 3, 15)
plotterf(np.angle(f_img_rot2), 'Phase of DFT of Rotated Image 2')

# P4: Extracting vectors of freq 0
f0_row = f_img[99, :]
f0_col = f_img[:, 99]
plt.figure('P4')
plt.suptitle('Zero Frequency Spectrum', color = 'white', backgroundcolor = 'green')
ax = plt.subplot(1, 2, 1)
plt.title('|F(0, v)|', color = 'white', backgroundcolor = 'green')
plt.plot(np.abs(f0_row))
ax = plt.subplot(1, 2, 2)
plt.title('|F(u, 0)|', color = 'white', backgroundcolor = 'green')
plt.plot(np.abs(f0_col))

# P5&6: Reading chest image
img_chest = cv.imread('chest.tif', cv.IMREAD_GRAYSCALE)
D = img_chest.dtype.itemsize * 8
rows, cols = img_chest.shape

C0 = 350
W = 100
P = 2*rows-1
Q = 2*cols-1

img_chest_padded = np.zeros((P, Q), np.uint8)
img_chest_padded[:rows, :cols] = img_chest

f_img_chest = np.fft.fft2(img_chest_padded)
f_img_chest = np.fft.fftshift(f_img_chest)

IBRF = np.ones_like(img_chest_padded)
for u in range(P):
    for v in range(Q):
        if C0-W/2 <= np.sqrt((u-P/2)**2 + (v-Q/2)**2) <= C0+W/2:
            IBRF[u, v] = 0

f_img_chest_filterd = f_img_chest * IBRF

# P7: Mirroring the image using fourier properties
f_img_chest_filterd_mirrored = np.conj(f_img_chest_filterd)

# P8: Plotting Input, Output, and the final freq domain
# Obtaining the corresponding filtered image
img_chest_filtered = np.fft.ifft2(np.fft.ifftshift(f_img_chest_filterd_mirrored)).real
img_chest_new = img_chest_filtered[rows:, cols:] # The indices needs to be flipped too
img_chest_new = contrast_stretching(img_chest_new.astype('uint8'))
# img_chest_new = np.clip(img_chest_new, 0, 255).astype('uint8')

plt.figure('P8')
ax = plt.subplot(1, 3, 1)
plotter(img_chest, 'Original Chest')
ax = plt.subplot(1, 3, 2)
plotter(img_chest_new, 'Filtered Chest')
ax = plt.subplot(1, 3, 3)
plotterf(np.log(1 + np.abs(f_img_chest_filterd)), 'Filtered Chest in freq domain')

plt.show()
