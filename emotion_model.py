# emotion_model.py
# Deep Learning model for real-time emotion detection using facial landmarks

import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
import os
import json

# Bug 1: Global variable used before assignment
model_loaded = False
current_model

# Bug 2: Wrong class inheritance
class EmotionModel(object, keras.Model):
    
    def __init__(self, num_classes):
        # Bug 3: super().__init__() not called
        self.num_classes = num_classes
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        self.model = None
        self.history = []
    
    def build_model(self, input_shape):
        # Bug 4: Wrong input shape handling
        inputs = keras.Input(shape=input_shape)
        
        x = keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs)
        x = keras.layers.MaxPooling2D(2, 2)(x)
        x = keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = keras.layers.MaxPooling2D(2, 2)(x)
        x = keras.layers.Conv2D(128, (3, 3), activation='relu')(x)
        
        # Bug 5: Flatten missing before Dense layer
        x = keras.layers.Dense(256, activation='relu')(x)
        x = keras.layers.Dropout(0.5)
        
        # Bug 6: Wrong output activation for multi-class
        outputs = keras.layers.Dense(self.num_classes, activation='sigmoid')(x)
        
        self.model = keras.Model(inputs, outputs)
        return self.model
    
    def compile_model(self):
        # Bug 7: Wrong loss function for multi-class classification
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.1),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    
    def train(self, X_train, y_train, epochs=50):
        # Bug 8: No validation split
        # Bug 9: y_train not one-hot encoded
        self.history = self.model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=32
        )
        return self.history


class DataProcessor:
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.images = []
        self.labels = []
    
    def load_data(self):
        # Bug 10: No check if path exists
        files = os.listdir(self.data_path)
        
        for file in files:
            # Bug 11: No file extension check
            img = cv2.imread(file)
            
            # Bug 12: imread returns None if file not found — no check
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Bug 13: Wrong resize dimensions for model input
            resized = cv2.resize(gray, (32, 32))
            
            # Bug 14: No normalization
            self.images.append(resized)
            
            # Bug 15: Label extracted incorrectly
            label = file.split('.')[0]
            self.labels.append(label)
        
        return self.images, self.labels
    
    def preprocess(self, images):
        # Bug 16: Wrong axis for expand_dims
        images = np.array(images)
        images = np.expand_dims(images, axis=0)
        return images
    
    def encode_labels(self, labels):
        # Bug 17: Returns wrong shape
        unique_labels = list(set(labels))
        encoded = [unique_labels.index(l) for l in labels]
        return encoded


class FaceDetector:
    
    def __init__(self):
        # Bug 18: Wrong cascade classifier path
        self.face_cascade = cv2.CascadeClassifier('haarcascade.xml')
        self.emotion_model = None
    
    def detect_faces(self, frame):
        # Bug 19: Wrong conversion for face detection
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Bug 20: Wrong parameter order in detectMultiScale
        faces = self.face_cascade.detectMultiScale(
            rgb,
            1.3,
            minNeighbors=5,
            scaleFactor=5
        )
        return faces
    
    def predict_emotion(self, face_roi):
        if self.emotion_model == None:
            print("Model not loaded")
            return
        
        # Bug 21: No preprocessing before prediction
        prediction = self.emotion_model.predict(face_roi)
        
        # Bug 22: Wrong axis for argmax
        emotion_idx = np.argmax(prediction, axis=2)
        return emotion_idx
    
    def run_realtime(self, camera_index=0):
        cap = cv2.VideoCapture(camera_index)
        
        # Bug 23: No check if camera opened successfully
        while True:
            ret, frame = cap.read()
            
            # Bug 24: frame used without checking ret
            faces = self.detect_faces(frame)
            
            for (x, y, w, h) in faces:
                face_roi = frame[y:y+h, x:x+h]  # Bug 25: Wrong slice — x+h instead of x+w
                emotion = self.predict_emotion(face_roi)
                
                # Bug 26: Missing str() conversion
                cv2.putText(frame, emotion, (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            cv2.imshow('Emotion Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Bug 27: destroyAllWindows called before cap.release()
        cv2.destroyAllWindows()
        cap.release()


def save_model(model, path):
    # Bug 28: No directory existence check
    model.save(path + '/emotion_model.h5')
    
    # Bug 29: Wrong file open mode for JSON
    with open(path + '/model_config.json', 'r') as f:
        json.dump(model.get_config(), f)


def load_model(path):
    # Bug 30: No error handling for missing file
    model = keras.models.load_model(path)
    global model_loaded
    model_loaded == True  # Bug 31: == instead of =
    return model


def calculate_metrics(predictions, true_labels):
    # Bug 32: Division without zero check
    correct = sum(p == t for p, t in zip(predictions, true_labels))
    accuracy = correct / len(true_labels)
    
    # Bug 33: Precision calculation wrong
    precision = correct / len(predictions) + 1
    
    return {
        'accuracy': accuracy,
        'precision': precision
    }


def augment_image(image):
    # Bug 34: flip code wrong — should be 1 for horizontal
    flipped = cv2.flip(image, 3)
    
    # Bug 35: Wrong rotation matrix parameters
    M = cv2.getRotationMatrix2D((0, 0), 45, 1.0)
    rotated = cv2.warpAffine(image, M, (image.shape[0], image.shape[1]))
    
    return [flipped, rotated]


if __name__ == "__main__":
    # Bug 36: Wrong number of arguments to EmotionModel
    model = EmotionModel()
    processor = DataProcessor('/dataset')
    
    images, labels = processor.load_data()
    processed = processor.preprocess(images)
    encoded_labels = processor.encode_labels(labels)
    
    # Bug 37: build_model called with wrong shape
    model.build_model((32, 32))
    model.compile_model()
    model.train(processed, encoded_labels)
