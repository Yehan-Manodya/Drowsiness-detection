from ultralytics import YOLO
import cv2
import numpy as np

class DrowsinessDetector:
    def __init__(self, model_path='models/best.pt'):
        print("Loading YOLOv8n model...")
        self.model = YOLO(model_path)
        self.confidence_threshold = 0.80  # 80% confidence
        print("Model loaded successfully!")

    def predict(self, frame):
        """
        Takes a frame and returns:
        - label: 'alert' or 'drowsy'
        - confidence: 0.0 to 1.0
        - is_confident: True if confidence > 80%
        """
        # Resize frame for model
        resized = cv2.resize(frame, (224, 224))

        # Run prediction
        results = self.model(resized, verbose=False)

        # Get probabilities
        probs = results[0].probs
        confidence = float(probs.top1conf)
        class_id = int(probs.top1)

        # Get class name
        label = self.model.names[class_id]

        # Check if confident enough
        is_confident = confidence >= self.confidence_threshold

        return {
            'label': label,
            'confidence': confidence,
            'is_confident': is_confident
        }