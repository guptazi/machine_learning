import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import os

# Define the base paths for the original and face directories
base_path_original = 'HW9- original'
base_path_faces = 'HW9-face'

# Load pre-trained Haar cascades face detector from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Iterate over each subfolder in the HW9-original directory
for person_folder in os.listdir(base_path_original):
    # Construct the path to the current person's folder
    person_path = os.path.join(base_path_original, person_folder)
    
    # Create the corresponding directory in HW9-face if it doesn't exist
    face_folder_path = os.path.join(base_path_faces, person_folder)
    if not os.path.exists(face_folder_path):
        os.makedirs(face_folder_path)
    
    # Iterate over each image in the current person's folder
    for img_file in os.listdir(person_path):
        # Construct the path to the image file
        img_path = os.path.join(person_path, img_file)

        # Load the image
        img = cv2.imread(img_path)

        # Convert the image to grayscale for face detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        # Process each detected face
        for idx, (x, y, w, h) in enumerate(faces):
            # Crop the face from the image
            face = img[y:y+h, x:x+w]
            # Convert face to PIL Image to resize
            face_image = Image.fromarray(face)
            face_image = face_image.resize((64, 96))
            # Construct the path to save the cropped face
            face_path = os.path.join(face_folder_path, f'{os.path.splitext(img_file)[0]}_face{idx}.jpg')
            # Save the resized face
            face_image.save(face_path)
