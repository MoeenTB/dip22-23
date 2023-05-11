import numpy as np

img = np.array([[0, 1, 4, 5, 5, 1, 10],
                [5, 7, 5, 6, 5, 6, 8],
                [3, 8, 0, 9, 10, 6, 0],
                [5, 0, 8, 8, 1, 0, 7],
                [4, 5, 0, 7, 0, 5, 0],
                [3, 2, 4, 5, 2, 10, 5],
                [1, 6, 9, 0, 6, 9, 8]])
print(img)
kernel = np.array([[0, 2, 0],
                   [2, 4, 2],
                   [0, 2, 0]])

kernel = kernel / np.sum(kernel)

size = 3
m = img.shape[0]
n = img.shape[1]
img_padded = np.zeros((m + size - 1, n + size - 1), dtype = img.dtype) # Zero padding
img_padded[int(size/2):int(size/2)+m, int(size/2):int(size/2)+n] = img

img_filtered = np.zeros_like(img, dtype = np.float64)
for i in range(m):
    for j in range(n):
        img_filtered[i, j] = np.sum(img_padded[i:i+size, j:j+size] * kernel)

print(np.round(img_filtered, 2))