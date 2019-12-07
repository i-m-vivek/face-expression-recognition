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

W = np.zeros((20, 20))

for i in range(20):
    for j in range(20):
        dist = (i-10)**2 + (j-10)**2
        W[i, j] = np.exp(-dist/100)
        
plt.imshow(W, cmap="gray")
plt.show()


out = convolve2d(bw, W)
plt.imshow(out, cmap = "gray")
plt.show()

print(bw.shape)
print(out.shape)

out = convolve2d(bw, W, mode="same")
plt.imshow(out, cmap = "gray")
plt.show()
print(out.shape)

#ndoing the blur on the third channel only then adding all the layers 
temp = convolve2d(img[:,:,2], W, mode="same")
newImg = img
newImg[:,:,2] = temp
plt.imshow(newImg)

temp = convolve2d(img[:,:,1], W, mode="same")
newImg = img
newImg[:,:,1] = temp
plt.imshow(newImg)


W/=W.sum()
temp = convolve2d(img[:,:,0], W, mode="same")
newImg = img
newImg[:,:,0] = temp
newImg[:,:,0] /= newImg[:,:,0].max()
plt.imshow(newImg)

