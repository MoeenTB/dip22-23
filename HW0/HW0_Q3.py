import numpy as np
import matplotlib.pyplot as plt

std_num = 401133010
r = 10

def circ_sqr(r : int):
    '''
    r is the radius of the circle
    '''
    # core = np.full((2*r-3, 2*r-3), 255)
    # sqr = np.zeros((2*r-1, 2*r-1))
    # sqr[1:-1, 1:-1] = core
    # sqr[0, r-1] = 255
    # sqr[r-1, 0] = 255
    # sqr[-1, r-1] = 255
    # sqr[r-1, -1] = 255
    # sqr = sqr.astype(np.uint8)
    sqr = np.zeros((2*r-1, 2*r-1), dtype = np.uint8)
    for i in range(0, 2*r-1):
        x = -abs(i-r+1) + (r-1)
        sqr[i, r-1-x:r-1+x+1] = 255
    return sqr

def add_noise(mat, A):
    '''
    Adds noise with amplitude A and uniform distribution to the elemenets of mat.
    '''
    noise = np.random.uniform(0, A, mat.shape)
    mask = (mat == 0)
    mask = 2*mask - 1
    mat = np.floor(mat + mask*noise).astype(np.uint8)
    # print(mat.dtype)
    return mat

s = 0
t = std_num
while (t > 0):
    s += t%10
    t //= 10
A = 20 + s%15

X = circ_sqr(r)
Y = add_noise(X, A)


ax = plt.subplot(1, 2, 1)
plt.suptitle(f'HW0-Image-{std_num}')
plt.title('Original Matrix')
plt.axis('off')
plt.imshow(X, cmap = 'gray')

plt.subplot(1, 2, 2)
plt.title(f'Noisy Matrix - Noise Amplitude = {A}')
plt.axis('off')
plt.imshow(Y, cmap = 'gray')

fig = plt.figure()
plt.suptitle(f'HW0-Surface-{std_num}')

ax = fig.add_subplot(1, 2, 1, projection = '3d')
x, y = np.indices(X.shape)
plt.title('Original Matrix in 3D')
# ax.axis('off')
ax.plot_wireframe(x, y, X, rstride = 1, cstride = 1)

ax = fig.add_subplot(1, 2, 2, projection = '3d')
x, y = np.indices(Y.shape)
plt.title('Noisy Matrix in 3D')
# ax.axis('off')
ax.plot_wireframe(x, y, Y, rstride = 1, cstride = 1)

plt.show()