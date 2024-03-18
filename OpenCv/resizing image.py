import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread(r'A0.png', 1)

resized = cv2.resize(image, (1000, 1000))

Titles =["Original", "Resized"]
images =[image, resized]
count = 2

for i in range(count):
	plt.subplot(2, 2, i + 1)
	plt.title(Titles[i])
	plt.imshow(images[i])

plt.show()
