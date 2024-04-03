import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA
import os

# Base directory for the edge-detected images
base_dir = 'HW9-edge'

# List to store file paths
filepaths = []

# Iterate over each subfolder and collect image file paths
for person_folder in os.listdir(base_dir):
    person_path = os.path.join(base_dir, person_folder)
    for img_file in os.listdir(person_path):
        filepaths.append(os.path.join(person_path, img_file))

# Load images, convert to grayscale if necessary, and flatten them
images = []
for filepath in filepaths:
    img = Image.open(filepath)
    # Convert to grayscale if the image is not already in grayscale
    if img.mode != 'L':
        img = img.convert('L')
    images.append(np.array(img).flatten())

# Perform PCA with 200 components
pca = PCA(n_components=100)
principalComponents = pca.fit_transform(images)

# Plot the principal components
plt.figure(figsize=(8, 6))
for i, _ in enumerate(principalComponents):
    plt.scatter(principalComponents[i, 0], principalComponents[i, 1], marker='o')
plt.title('PCA of Edge-Detected Images')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.show()
