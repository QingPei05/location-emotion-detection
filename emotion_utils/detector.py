from deepface import DeepFace
import cv2
import numpy as np
from emotion_utils.config import get_config

class EmotionDetector:
    def __init__(self):
        self.color_map = get_config()["color_map"]

    def detect_emotions(self, img):
        """Detect emotions using DeepFace"""
        try:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = DeepFace.analyze(
                img_path=img_rgb,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend='opencv',
                silent=True
            )
            
            detections = []
            for result in results:
                detections.append({
                    "emotion": result['dominant_emotion'],
                    "confidence": round(result['emotion'][result['dominant_emotion']], 2),
                    "x": result['region']['x'],
                    "y": result['region']['y'],
                    "w": result['region']['w'],
                    "h": result['region']['h']
                })
            return detections
        except Exception as e:
            print(f"Detection error: {e}")
            return []

    def draw_detections(self, img, detections):
        """Draw detection boxes with labels"""
        output_img = img.copy()
        for det in detections:
            x, y, w, h = det["x"], det["y"], det["w"], det["h"]
            emotion = det["emotion"]
            confidence = det["confidence"]
            color = self.color_map.get(emotion.lower(), (255, 255, 255))
            
            # Draw rectangle
            cv2.rectangle(output_img, (x, y), (x+w, y+h), color, 3)
            
            # Draw label
            label = f"{emotion} {confidence}%"
            cv2.putText(
                output_img, label,
                (x+5, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, color, 2
            )
        return output_img
