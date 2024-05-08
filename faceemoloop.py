import os
import cv2
import numpy as np
import time
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from datetime import datetime
import csv

face_classifier = cv2.CascadeClassifier(r'E:\\GAN for Face expression Classification\\final deployment\\haarcascade_frontalface_default.xml')
classifier = load_model(r'E:\\GAN for Face expression Classification\\final deployment\\model.h5')

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

cap = cv2.VideoCapture(0)

# Create a directory to save the images
if not os.path.exists('face_images'):
    os.makedirs('face_images')

# Open CSV file to write emotions
with open('face_emotions.csv', mode='w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['time', 'face_emotion'])

start_time = time.time()

for i in range(4000):  # Run 30 times
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            prediction = classifier.predict(roi)[0]
            label = emotion_labels[prediction.argmax()]

            # Save the image
            cv2.imwrite(f'face_images/image_{i}.jpg', roi_gray)

            # Write emotion to CSV
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            with open('face_emotions.csv', mode='a', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([timestamp, label])

            # Draw rectangle and label on the frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            label_position = (x, y)
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Emotion Detector', frame)
    if cv2.waitKey(1000) & 0xFF == ord('q') or time.time() - start_time >= 30:
        break

cap.release()
cv2.destroyAllWindows()
