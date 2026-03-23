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
current_model = None

# Bug 2: Wrong class inheritance
class EmotionModel(keras.Model):
    
    def __init__(self, num_classes):
        # Bug 3: super().__init__() not called
        super().__init__()
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
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(256, activation='relu')(x)
        x = keras.layers.Dropout(0.5)(x)
        
        # Bug 6: Wrong output activation for multi-class
        outputs = keras.layers.Dense(self.num_classes, activation='softmax')(x)
        
        self.model = keras.Model(inputs, outputs)
        return self.model
    
    def compile_model(self):
        # Bug 7: Wrong loss function for multi-class classification
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def train(self, X_train, y_train, epochs=50):
        # Bug 8: No validation split
        # Bug 9: y_train not one-hot encoded
        y_train_encoded = keras.utils.to_categorical(y_train, num_classes=self.num_classes)
        self.history = self.model.fit(
            X_train,
            y_train_encoded,
            epochs=epochs,
            batch_size=32,
            validation_split=0.2
        )
        return self.history


class DataProcessor:
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.images = []
        self.labels = []
    
    def load_data(self):
        # Bug 10: No check if path exists
        if not os.path.exists(self.data_path):
            raise ValueError(f"Path {self.data_path} does not exist")
        
        files = os.listdir(self.data_path)
        
        for file in files:
            # Bug 11: No file extension check
            if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            
            file_path = os.path.join(self.data_path, file)
            img = cv2.imread(file_path)
            
            # Bug 12: imread returns None if file not found — no check
            if img is None:
                continue
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Bug 13: Wrong resize dimensions for model input
            resized = cv2.resize(gray, (48, 48))
            
            # Bug 14: No normalization
            normalized = resized / 255.0
            self.images.append(normalized)
            
            # Bug 15: Label extracted incorrectly
            label = file.split('_')[0]  # Assuming format: emotion_id.jpg
            self.labels.append(label)
        
        return self.images, self.labels
    
    def preprocess(self, images):
        # Bug 16: Wrong axis for expand_dims
        images = np.array(images)
        images = np.expand_dims(images, axis=-1)
        return images
    
    def encode_labels(self, labels):
        # Bug 17: Returns wrong shape
        unique_labels = list(set(labels))
        encoded = [unique_labels.index(l) for l in labels]
        return np.array(encoded)


class FaceDetector:
    
    def __init__(self):
        # Bug 18: Wrong cascade classifier path
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.emotion_model = None
    
    def detect_faces(self, frame):
        # Bug 19: Wrong conversion for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Bug 20: Wrong parameter order in detectMultiScale
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5
        )
        return faces
    
    def predict_emotion(self, face_roi):
        if self.emotion_model == None:
            print("Model not loaded")
            return "unknown"
        
        # Bug 21: No preprocessing before prediction
        face_roi = cv2.resize(face_roi, (48, 48))
        face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        face_roi = face_roi / 255.0
        face_roi = np.expand_dims(face_roi, axis=0)
        face_roi = np.expand_dims(face_roi, axis=-1)
        
        prediction = self.emotion_model.predict(face_roi)
        
        # Bug 22: Wrong axis for argmax
        emotion_idx = np.argmax(prediction, axis=1)[0]
        emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        return emotion_labels[emotion_idx]
    
    def run_realtime(self, camera_index=0):
        cap = cv2.VideoCapture(camera_index)
        
        # Bug 23: No check if camera opened successfully
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        while True:
            ret, frame = cap.read()
            
            # Bug 24: frame used without checking ret
            if not ret:
                print("Error: Could not read frame")
                break
            
            faces = self.detect_faces(frame)
            
            for (x, y, w, h) in faces:
                face_roi = frame[y:y+h, x:x+w]  # Bug 25: Wrong slice — x+h instead of x+w
                emotion = self.predict_emotion(face_roi)
                
                # Bug 26: Missing str() conversion
                cv2.putText(frame, str(emotion), (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            cv2.imshow('Emotion Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Bug 27: destroyAllWindows called before cap.release()
        cap.release()
        cv2.destroyAllWindows()


def save_model(model, path):
    # Bug 28: No directory existence check
    os.makedirs(path, exist_ok=True)
    model.save(path + '/emotion_model.h5')
    
    # Bug 29: Wrong file open mode for JSON
    with open(path + '/model_config.json', 'w') as f:
        json.dump(model.get_config(), f)


def load_model(path):
    # Bug 30: No error handling for missing file
    try:
        model = keras.models.load_model(path)
        global model_loaded
        model_loaded = True  # Bug 31: == instead of =
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def calculate_metrics(predictions, true_labels):
    # Bug 32: Division without zero check
    if len(true_labels) == 0:
        return {'accuracy': 0, 'precision': 0}
    
    correct = sum(p == t for p, t in zip(predictions, true_labels))
    accuracy = correct / len(true_labels)
    
    # Bug 33: Precision calculation wrong
    precision = correct / len(true_labels) if len(true_labels) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision
    }


def augment_image(image):
    # Bug 34: flip code wrong — should be 1 for horizontal
    flipped = cv2.flip(image, 1)
    
    # Bug 35: Wrong rotation matrix parameters
    center = (image.shape[1]//2, image.shape[0]//2)
    M = cv2.getRotationMatrix2D(center, 45, 1.0)
    rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    
    return [flipped, rotated]


if __name__ == "__main__":
    # Bug 36: Wrong number of arguments to EmotionModel
    model = EmotionModel(7)
    processor = DataProcessor('/dataset')
    
    images, labels = processor.load_data()
    processed = processor.preprocess(images)
    encoded_labels = processor.encode_labels(labels)
    
    # Bug 37: build_model called with wrong shape
    model.build_model((48, 48, 1))
    model.compile_model()
    model.train(processed, encoded_labels)