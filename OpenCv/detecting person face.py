import cv2
import numpy as np

# Load pre-trained face detection model
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the face recognizer (using LBPH Face Recognizer here)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trained_model.yml') # Load your trained model

# Start the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Predict the identity of the face
        id_, confidence = recognizer.predict(gray[y:y+h, x:x+w])

        # Check if the face is one of the three individuals (assuming their labels are 1, 2, and 3)
        if id_ in [1, 2, 3] and confidence < 50: # Adjust confidence as needed
            print(f"Recognized ID: {id_} with confidence {confidence}")

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
