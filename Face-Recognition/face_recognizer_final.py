import cv2
import os
import numpy as np
from datetime import datetime
import csv

face_model = cv2.face.LBPHFaceRecognizer_create()


data_dir = r"D:\Semester 2 files\Fall detection project\Face detector\face recognition final\Frames_face\1.VISHAL"

width = 24
height = 24

samples = []
labels = []
names = ["VISHAL"]

students = names.copy()

for filename in os.listdir(data_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = os.path.join(data_dir, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (width, height))
        samples.append(image)
        labels.append(0) 

samples = np.array(samples, dtype=np.uint8)
labels = np.array(labels, dtype=np.int32)

face_model.train(samples, labels)

face_model.save('trained_model.xml')

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

video_capture = cv2.VideoCapture(0)

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

f = open(current_date+'.csv','w+',newline = '')
lnwriter = csv.writer(f)

while True:
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_roi = gray[y:y + h, x:x + w]

        label_id, confidence = face_model.predict(face_roi)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        print(confidence)
        if confidence < 213.999999999999 and label_id == 0:  
            name = names[label_id]
            if name in students:
                    students.remove(name)
                    print(students)
                    current_time = now.strftime("%H-%M-%S")
                    lnwriter.writerow([name,current_time])
        else:
            name = 'Unknown'
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()

cv2.destroyAllWindows()
