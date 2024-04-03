from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
from PIL import Image
import os

# Initialize variables
images = []
labels = []

# Base directory for the processed images
base_dir = 'HW9-edge'

# Iterate over each subfolder and collect image file paths
for i, person_folder in enumerate(sorted(os.listdir(base_dir))):
    person_path = os.path.join(base_dir, person_folder)
    for img_file in os.listdir(person_path):
        # Construct the full file path
        filepath = os.path.join(person_path, img_file)
        # Load the image, convert to grayscale if necessary, and flatten it
        img = Image.open(filepath)
        if img.mode != 'L':
            img = img.convert('L')
        images.append(np.array(img).flatten())
        # Assign labels based on the person_folder name
        labels.append(i)

# Convert the lists to numpy arrays for processing with sklearn
X = np.array(images)
y = np.array(labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the SVC model
svc = SVC(kernel='linear')  # You can experiment with different kernels

# Train the model
svc.fit(X_train, y_train)

# Predict on the test set
y_pred = svc.predict(X_test)

print(y_pred)
