import os
from dataclasses import dataclass

@dataclass
class AppSettings:
    # Detection Settings
    MIN_CONFIDENCE: float = 0.65
    DETECTOR_BACKENDS: list = ["mtcnn", "retinaface", "opencv", "ssd", "dlib"]
    DEFAULT_DETECTOR: str = "mtcnn"
    ENSEMBLE_MODELS: list = ["VGG-Face", "Facenet", "OpenFace"]
    
    # UI Settings
    DEFAULT_LANGUAGE: str = "English"
    AVAILABLE_LANGS: list = ["English", "中文", "Malay"]
    
    # Data Settings
    DATA_DIR: str = "data"
    HISTORY_FILE: str = os.path.join(DATA_DIR, "history.csv")
    MAX_HISTORY_RECORDS: int = 1000
    
    # Performance
    CACHE_EXPIRE: int = 3600  # 1 hour

settings = AppSettings()
