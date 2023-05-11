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

# P1: Reading chest image
img = cv.imread('shoulder.jpg', cv.IMREAD_GRAYSCALE)
print("Dimensions of the original chest image:", img.shape)
D = img.dtype.itemsize * 8
# rows, cols = img.shape

# P2: Defining the filtering function
def my_filter(img:np.ndarray, ftype:list, params:dict):
    rows, cols = img.shape
    # Padding the image for periodic convolution
    P = 2*rows-1
    Q = 2*cols-1
    img_p = np.zeros((P, Q), np.uint8)
    img_p[:rows, :cols] = img

    f_img_p = np.fft.fft2(img_p)
    f_img_p = np.fft.fftshift(f_img_p)

    H = np.zeros_like(img_p, np.float64)
    if ftype[0] == 'ideal':
        D0 = params['D0']
        for u in range(P):
            for v in range(Q):
                if np.sqrt((u-P/2)**2 + (v-Q/2)**2) <= D0:
                    H[u, v] = 1
    elif ftype[0] == 'butter':
        D0 = params['D0']
        n = params['n']
        for u in range(P):
            for v in range(Q):
                H[u, v] = 1/(1+np.power(np.sqrt((u-P/2)**2 + (v-Q/2)**2)/D0, 2*n))
    elif ftype[0] == 'gaussian':
        D0 = params['D0']
        for u in range(P):
            for v in range(Q):
                H[u, v] = np.exp(-((u-P/2)**2 + (v-Q/2)**2)/(2*D0**2))
    else:
        print("Invalid Type")
    if ftype[1] == 'H':
        H = 1 - H
    f_img_p_filtered = f_img_p * H
    img_p_filtered = np.fft.ifft2(np.fft.ifftshift(f_img_p_filtered)).real
    img_filtered = img_p_filtered[:rows, :cols]
    # print(img_chest_new.shape)
    # img_filtered = contrast_stretching(img_filtered.astype('uint8'))
    img_chest_new = np.clip(img_filtered, 0, 255).astype('uint8')
    return img_filtered

radii = np.array([50, 100, 200])
plt.figure('LPF')
for i, radius in enumerate(radii):
    img_filtered = my_filter(img, ['ideal', 'L'], {'D0':radius})
    ax = plt.subplot(3, 3, 3*i+1)
    plotterf(img_filtered, f'ILPF w/ D0 ={radius}')
    img_filtered = my_filter(img, ['butter', 'L'], {'D0':radius, 'n':2})
    ax = plt.subplot(3, 3, 3*i+2)
    plotterf(img_filtered, f'BLPF w/ D0 ={radius}')
    img_filtered = my_filter(img, ['gaussian', 'L'], {'D0':radius})
    ax = plt.subplot(3, 3, 3*i+3)
    plotterf(img_filtered, f'GLPF w/ D0 ={radius}')

plt.figure('HPF')
for i, radius in enumerate(radii):
    img_filtered = my_filter(img, ['ideal', 'H'], {'D0':radius})
    ax = plt.subplot(3, 3, 3*i+1)
    plotterf(img_filtered, f'IHPF w/ D0 ={radius}')
    img_filtered = my_filter(img, ['butter', 'H'], {'D0':radius, 'n':2})
    ax = plt.subplot(3, 3, 3*i+2)
    plotterf(img_filtered, f'BHPF w/ D0 ={radius}')
    img_filtered = my_filter(img, ['gaussian', 'H'], {'D0':radius})
    ax = plt.subplot(3, 3, 3*i+3)
    plotterf(img_filtered, f'GHPF w/ D0 ={radius}')

plt.show()
