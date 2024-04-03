import cv2
import os

# Define the base paths for the face and edge directories
base_path_faces = 'HW9-face'
base_path_edges = 'HW9-edge'

# Thresholds for Canny edge detection
t_lower = 50
t_upper = 150

# Iterate over each subfolder in the HW9-face directory
for person_folder in os.listdir(base_path_faces):
    # Construct the path to the current person's folder
    person_path = os.path.join(base_path_faces, person_folder)
    
    # Create the corresponding directory in HW9-edge if it doesn't exist
    edge_folder_path = os.path.join(base_path_edges, person_folder)
    if not os.path.exists(edge_folder_path):
        os.makedirs(edge_folder_path)
    
    # Iterate over each image in the current person's folder
    for img_file in os.listdir(person_path):
        # Construct the path to the image file
        img_path = os.path.join(person_path, img_file)

        # Load the image
        img = cv2.imread(img_path)

        # Apply Canny edge detection
        edge = cv2.Canny(img, t_lower, t_upper)

        # Construct the path to save the edge-detected image
        edge_path = os.path.join(edge_folder_path, img_file)

        # Save the edge-detected image
        cv2.imwrite(edge_path, edge)
