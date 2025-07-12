from deepface import DeepFace
import cv2
import numpy as np
from typing import List, Dict

class EmotionDetector:
    def __init__(self):
        # Color mapping for visualization (BGR format)
        # Maintained original colors as requested
        self.color_map = {
            "happy": (0, 255, 0),      # Green
            "neutral": (255, 255, 0),   # Yellow
            "sad": (0, 0, 255),        # Red
            "angry": (0, 165, 255),    # Orange
            "fear": (128, 0, 128),     # Purple
            "surprise": (255, 0, 255), # Pink
            "disgust": (0, 128, 0)     # Dark Green
        }
        
        # Emotion priority mapping for handling ambiguous cases
        # Lower numbers = higher priority when conflicts occur
        self.emotion_priority = {
            "angry": 1,    # Highest priority (most important to detect correctly)
            "fear": 2,
            "disgust": 3,
            "sad": 4,
            "surprise": 5,
            "happy": 6,
            "neutral": 7    # Lowest priority
        }

    def detect_emotions(self, img: np.ndarray) -> List[Dict]:
        """Enhanced emotion detection with improved accuracy
        
        Args:
            img: Input image in BGR format (OpenCV default)
            
        Returns:
            List of detection dictionaries containing:
            - emotion: dominant emotion label
            - confidence: detection confidence (0-100)
            - x, y, w, h: bounding box coordinates
        """
        try:
            # --- Preprocessing ---
            # Convert to RGB (DeepFace expects RGB)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Resize to fixed dimensions for consistent detection
            img_resized = cv2.resize(img_rgb, (640, 480))  
            
            # --- Emotion Analysis ---
            # Using retinaface backend for better accuracy
            results = DeepFace.analyze(
                img_path=img_resized,
                actions=['emotion'],    # Focus only on emotion
                enforce_detection=True,  # Strict mode - no silent failures
                detector_backend='retinaface',  # Most accurate detector
                silent=True              # Suppress verbose output
            )
            
            # --- Post-processing ---
            detections = []
            for result in results:
                dominant_emotion = result['dominant_emotion']
                confidence = round(result['emotion'][dominant_emotion], 2)
                
                # Confidence thresholding
                if confidence < 50:  # Low confidence -> fallback to neutral
                    dominant_emotion = "neutral"
                    confidence = max(result['emotion'].values())
                
                # Special handling for easily confused emotions
                if dominant_emotion in ["angry", "disgust"] and confidence < 70:
                    # Pick the one with higher confidence
                    dominant_emotion = max(
                        ["angry", "disgust"],
                        key=lambda x: result['emotion'][x]
                    )
                
                # Package detection results
                detections.append({
                    "emotion": dominant_emotion,
                    "confidence": confidence,
                    "x": result['region']['x'],
                    "y": result['region']['y'],
                    "w": result['region']['w'],
                    "h": result['region']['h']
                })
            
            # Sort by emotion priority (important emotions first)
            detections.sort(key=lambda x: self.emotion_priority.get(x["emotion"], 8))
            return detections
            
        except Exception as e:
            print(f"Detection error: {str(e)}")
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
