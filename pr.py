import cv2
from keras.models import load_model
import numpy as np

# Load the pre-trained face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + r'C:\Users\kushw\PycharmProjects\EMOTION\haarcascade_frontalface_default.xml')

# Load the pre-trained emotion detection model
emotion_model = load_model(r'C:\Users\kushw\PycharmProjects\EMOTION\monu_cnn.h5')

# Define the emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Function to detect and classify emotions in faces
def detect_emotions(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (48, 48))
        face_roi = np.expand_dims(face_roi, axis=0)
        face_roi = np.expand_dims(face_roi, axis=3)

        emotion_prediction = emotion_model.predict(face_roi)
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion = emotion_labels[emotion_label_arg]

        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

    return image

# Load an image
image_path = 'image.jpg'
image = cv2.imread(image_path)

# Detect emotions in the image
image_with_emotions = detect_emotions(image)

# Display the image with emotions
cv2.imshow('Emotion Detection', image_with_emotions)
cv2.waitKey(0)
cv2.destroyAllWindows()
