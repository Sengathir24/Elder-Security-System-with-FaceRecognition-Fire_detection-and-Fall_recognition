import cv2
import os
import numpy as np
from datetime import datetime
import csv

# Set up the LBPH face recognizer
face_model = cv2.face.LBPHFaceRecognizer_create()

# Path to the directory containing the face dataset for the person
data_dir = r"D:\Semester 2 files\Fall detection project\Face detector\face recognition final\Frames_face\1.VISHAL"

# Size of the face images
width = 24
height = 24

# Create a list to store the face samples and corresponding labels
samples = []
labels = []
names = ["VISHAL"]

students = names.copy()

# Load face samples from the dataset for the person
for filename in os.listdir(data_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = os.path.join(data_dir, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (width, height))
        samples.append(image)
        labels.append(0)  # Assign label 0 to the person

# Convert the samples and labels to numpy arrays
samples = np.array(samples, dtype=np.uint8)
labels = np.array(labels, dtype=np.int32)

# Train the face recognition model
face_model.train(samples, labels)

# Save the trained model as an XML file
face_model.save('trained_model.xml')

# Load the Haar cascade classifier XML file
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize the video capture
video_capture = cv2.VideoCapture(0)

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")
 
 
 
f = open(current_date+'.csv','w+',newline = '')
lnwriter = csv.writer(f)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Process each detected face
    for (x, y, w, h) in faces:
        # Extract the face region of interest
        face_roi = gray[y:y + h, x:x + w]

        # Perform face recognition using the trained model
        label_id, confidence = face_model.predict(face_roi)

        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        print(confidence)
        # Display the person's name
        if confidence < 213.999999999999 and label_id == 0:  # Assuming label 0 corresponds to the trained person
            name = names[label_id]
            if name in students:
                    students.remove(name)
                    print(students)
                    current_time = now.strftime("%H-%M-%S")
                    lnwriter.writerow([name,current_time])
        else:
            name = 'Unknown'
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture
video_capture.release()

# Destroy all OpenCV windows
cv2.destroyAllWindows()
