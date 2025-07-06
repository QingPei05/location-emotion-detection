import cv2
import numpy as np
from deepface import DeepFace
from deepface.basemodels import VGGFace, OpenFace, Facenet, FbDeepFace
from typing import List, Dict, Optional

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
        self.min_confidence = 0.65  # 置信度阈值

    def _preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """图像预处理增强"""
        # 转换为灰度图进行直方图均衡
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        # 转换回BGR并应用CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        # 高斯模糊降噪
        img = cv2.GaussianBlur(img, (3, 3), 0)
        return img

    def _ensemble_detect(self, img_rgb: np.ndarray) -> List[Dict]:
        """集成多个模型进行检测"""
        models = ["VGG-Face", "Facenet", "OpenFace", "DeepFace"]
        results = []
        
        for model_name in models:
            try:
                result = DeepFace.analyze(
                    img_path=img_rgb,
                    actions=['emotion'],
                    enforce_detection=False,
                    detector_backend="mtcnn",  # 使用更准确的MTCNN检测器
                    model_name=model_name,
                    silent=True
                )
                if isinstance(result, list):
                    results.extend(result)
            except Exception as e:
                print(f"Model {model_name} error: {str(e)}")
                continue
                
        return results

    def detect_emotions(self, img: np.ndarray) -> List[Dict]:
        """增强的情绪检测方法"""
        try:
            # 图像预处理
            img = self._preprocess_image(img)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 集成模型检测
            raw_results = self._ensemble_detect(img_rgb)
            if not raw_results:
                return []

            # 结果聚合
            final_detections = []
            for result in raw_results:
                emotion = result['dominant_emotion']
                confidence = result['emotion'][emotion]
                
                # 过滤低置信度结果
                if confidence < self.min_confidence:
                    continue
                    
                final_detections.append({
                    "emotion": emotion,
                    "confidence": round(confidence * 100, 2),
                    "x": result['region']['x'],
                    "y": result['region']['y'],
                    "w": result['region']['w'],
                    "h": result['region']['h'],
                    "face_confidence": round(result['face_confidence'], 2)
                })

            return final_detections
        except Exception as e:
            print(f"Detection error: {str(e)}")
            return []

    def draw_detections(self, img: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """绘制检测结果"""
        output_img = img.copy()
        for det in detections:
            x, y, w, h = det["x"], det["y"], det["w"], det["h"]
            emotion = det["emotion"]
            confidence = det["confidence"]
            color = self.color_map.get(emotion.lower(), (255, 255, 255))
            
            # 绘制人脸框
            cv2.rectangle(output_img, (x, y), (x+w, y+h), color, 2)
            
            # 绘制信息文本
            label = f"{emotion} {confidence}%"
            cv2.putText(output_img, label, (x+5, y-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # 添加人脸置信度
            face_conf = f"Face: {det['face_confidence']}%"
            cv2.putText(output_img, face_conf, (x+5, y+h+25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        
        return output_img
