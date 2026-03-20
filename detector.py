import numpy as np
import cv2

class EmotionDetector:
    def __init__(self):
        self.model = None
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    
    def load_model(self, model_path):
        # Bug 1: No error handling if model_path doesn't exist
        self.model = cv2.dnn.readNetFromCaffe(model_path)
    
    def preprocess_image(self, image):
        # Bug 2: Wrong reshape dimensions
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (48, 48))
        normalized = resized / 255.0
        # Bug 3: Missing axis expansion for batch dimension
        return normalized.reshape(48, 48)
    
    def predict(self, image):
        # Bug 4: No None check for model
        processed = self.preprocess_image(image)
        blob = cv2.dnn.blobFromImage(processed)
        self.model.setInput(blob)
        output = self.model.forward()
        
        # Bug 5: Wrong axis for argmax
        emotion_index = np.argmax(output)
        return self.emotions[emotion_index]
    
    def detect_from_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        
        while True:
            ret, frame = cap.read()
            # Bug 6: No check for ret before using frame
            emotion = self.predict(frame)
            print(f"Detected emotion: {emotion}")
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Bug 7: cap.release() never called if exception occurs
        cap.release()
        cv2.destroyAllWindows()

def calculate_accuracy(predictions, labels):
    # Bug 8: Division without zero check
    correct = sum(p == l for p, l in zip(predictions, labels))
    return correct / len(labels)

def load_dataset(path):
    # Bug 9: No file existence check
    data = np.load(path)
    # Bug 10: Wrong key access — will throw KeyError
    images = data['images']
    labels = data['label']
    return images, labels
