import cv2
from matplotlib import pyplot as plt
import numpy as np


img = cv2.imread('Images/parking-lot.jpg', cv2.IMREAD_GRAYSCALE)

ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)


edges = cv2.Canny(thresh1, 50, 150, apertureSize=3)


linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)


cdstP = cv2.cvtColor(thresh1, cv2.COLOR_GRAY2BGR)

if linesP is not None:
    for i in range(0, len(linesP)):
        l = linesP[i][0]
        cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0, 255, 0), 2, cv2.LINE_AA)


parking_spaces = [] 

# Visualization of the results
plt.figure(figsize=(10, 10))

# Original Image with Threshold
plt.subplot(131)
plt.imshow(thresh1, cmap='gray')
plt.title('Binarized Image')
plt.xticks([]), plt.yticks([])

# Edges Image
plt.subplot(132)
plt.imshow(edges, cmap='gray')
plt.title('Edge Detection')
plt.xticks([]), plt.yticks([])

# Detected Lines 
plt.subplot(133)
plt.imshow(cdstP)
plt.title('Detected Lines')
plt.xticks([]), plt.yticks([])

plt.show()
