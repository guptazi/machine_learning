import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np

img1Color = cv.imread('Images/faces/face3.png')  # queryImage
img2Color = cv.imread('Images\images.png')  # trainImage
b_channel, g_channel, r_channel = cv.split(img1Color)
img1_RGB = cv.merge((r_channel, g_channel, b_channel))
b_channel, g_channel, r_channel = cv.split(img2Color)
img2_RGB = cv.merge((r_channel, g_channel, b_channel))
img1 = cv.cvtColor(img1_RGB, cv.COLOR_BGR2GRAY)
img2 = cv.cvtColor(img2_RGB, cv.COLOR_BGR2GRAY)
sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
bf = cv.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
        
img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.figure(figsize=(30, 30))
plt.subplot(131); plt.imshow(img1_RGB)
plt.title('Query Image'); plt.xticks([]), plt.yticks([])
plt.subplot(132); plt.imshow(img2_RGB )
plt.title('Train Image'); plt.xticks([]), plt.yticks([])
plt.subplot(133); plt.imshow(img3)
plt.title('Matched Image'); plt.xticks([]), plt.yticks([])
plt.show()
