import cv2
import numpy as np
from typing import List, Dict, Optional
from deepface import DeepFace
from deepface.basemodels import VGGFace, Facenet, OpenFace
from tqdm import tqdm
import concurrent.futures
from configs.settings import settings

class EnhancedEmotionDetector:
    def __init__(self):
        self.color_map = {
            "happy": (0, 255, 0),      # Green
            "neutral": (255, 255, 0),  # Yellow
            "sad": (0, 0, 255),       # Red
            "angry": (0, 165, 255),   # Orange
            "fear": (128, 0, 128),    # Purple
            "surprise": (255, 0, 255),# Pink
            "disgust": (0, 128, 0)    # Dark Green
        }
        self.min_confidence = settings.MIN_CONFIDENCE
        self.detector_backend = settings.DEFAULT_DETECTOR
        self.models = settings.ENSEMBLE_MODELS

    def _preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """Enhanced image preprocessing"""
        # Convert to LAB color space for better lighting normalization
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # CLAHE on L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge channels and convert back to BGR
        lab = cv2.merge((l, a, b))
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Denoising
        img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        return img

    def _analyze_with_model(self, img_rgb: np.ndarray, model_name: str) -> Optional[Dict]:
        """Analyze image with a single model"""
        try:
            result = DeepFace.analyze(
                img_path=img_rgb,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend=self.detector_backend,
                model_name=model_name,
                silent=True
            )
            return result[0] if isinstance(result, list) else result
        except Exception:
            return None

    def _ensemble_detect(self, img_rgb: np.ndarray) -> List[Dict]:
        """Parallel model ensemble detection"""
        results = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(self._analyze_with_model, img_rgb, model): model 
                for model in self.models
            }
            
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)
        return results

    def _aggregate_results(self, raw_results: List[Dict]) -> List[Dict]:
        """Aggregate results from multiple models"""
        if not raw_results:
            return []

        # Voting system for emotion
        emotion_votes = {}
        for result in raw_results:
            emotion = result['dominant_emotion']
            confidence = result['emotion'][emotion]
            
            if confidence >= self.min_confidence:
                if emotion not in emotion_votes:
                    emotion_votes[emotion] = {
                        'count': 0,
                        'total_confidence': 0,
                        'region': result['region'],
                        'face_confidence': result['face_confidence']
                    }
                emotion_votes[emotion]['count'] += 1
                emotion_votes[emotion]['total_confidence'] += confidence

        # Select emotion with most votes, then highest confidence
        final_results = []
        for emotion, data in emotion_votes.items():
            avg_confidence = data['total_confidence'] / data['count']
            final_results.append({
                'emotion': emotion,
                'confidence': avg_confidence,
                'region': data['region'],
                'face_confidence': data['face_confidence']
            })

        return sorted(final_results, key=lambda x: (-x['confidence'], -x['face_confidence']))

    def detect_emotions(self, img: np.ndarray) -> List[Dict]:
        """Main detection method with enhanced pipeline"""
        try:
            # Preprocessing
            img = self._preprocess_image(img)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Ensemble detection
            raw_results = self._ensemble_detect(img_rgb)
            if not raw_results:
                return []

            # Result aggregation
            aggregated = self._aggregate_results(raw_results)
            
            # Format final results
            final_detections = []
            for result in aggregated:
                final_detections.append({
                    "emotion": result['emotion'],
                    "confidence": round(result['confidence'] * 100, 2),
                    "face_confidence": round(result['face_confidence'] * 100, 2),
                    "x": result['region']['x'],
                    "y": result['region']['y'],
                    "w": result['region']['w'],
                    "h": result['region']['h']
                })

            return final_detections
        except Exception as e:
            print(f"Detection error: {str(e)}")
            return []

    def draw_detections(self, img: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Enhanced visualization with more info"""
        output_img = img.copy()
        for det in detections:
            x, y, w, h = det["x"], det["y"], det["w"], det["h"]
            emotion = det["emotion"]
            confidence = det["confidence"]
            face_conf = det["face_confidence"]
            color = self.color_map.get(emotion.lower(), (255, 255, 255))
            
            # Draw face rectangle
            cv2.rectangle(output_img, (x, y), (x+w, y+h), color, 2)
            
            # Draw emotion label
            emotion_text = f"{emotion} {confidence}%"
            cv2.putText(output_img, emotion_text, (x+5, y-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Draw face confidence
            face_text = f"Face: {face_conf}%"
            cv2.putText(output_img, face_text, (x+5, y+h+25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            
            # Draw model info
            model_text = f"Models: {len(self.models)}"
            cv2.putText(output_img, model_text, (x+5, y+h+50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        
        return output_img
