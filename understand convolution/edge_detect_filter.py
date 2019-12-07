import numpy as np 
from scipy.signal import convolve2d
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg

img = mpimg.imread("lena.png")
plt.imshow(img)
plt.show()

bw = img.mean(axis = 2)
plt.imshow(bw, cmap = "gray")
plt.show()

hx = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]], dtype = np.float32 )

hy = hx.T

gx = convolve2d(bw, hx)
plt.imshow(gx, cmap ="gray")
gy = convolve2d(bw, hy)
plt.imshow(gy, cmap ="gray")

G = np.sqrt(gx*gx + gy*gy)
plt.imshow(G, cmap ="gray")
#g = gx + gy
#plt.imshow(g, cmap ="gray")



x = 10
y = -10
hx = np.array([
        [x, x, x],
        [0, 0, 0],
        [y, y, y]], dtype = np.float32 )

hy = hx.T

gx = convolve2d(bw, hx)
plt.imshow(gx, cmap ="gray")
plt.show()
gy = convolve2d(bw, hy)
plt.imshow(gy, cmap ="gray")
plt.show()
g = gx + gy
plt.imshow(g, cmap ="gray")
plt.show()