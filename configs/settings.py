import os
from dataclasses import dataclass, field
from typing import List

@dataclass
class AppSettings:
    # Detection Settings
    MIN_CONFIDENCE: float = 0.65
    DETECTOR_BACKENDS: List[str] = field(default_factory=lambda: ["mtcnn", "retinaface", "opencv", "ssd", "dlib"])
    DEFAULT_DETECTOR: str = "mtcnn"
    ENSEMBLE_MODELS: List[str] = field(default_factory=lambda: ["VGG-Face", "Facenet", "OpenFace"])
    
    # UI Settings
    DEFAULT_LANGUAGE: str = "English"
    AVAILABLE_LANGS: List[str] = field(default_factory=lambda: ["English", "中文", "Malay"])
    
    # Data Settings
    DATA_DIR: str = "data"
    HISTORY_FILE: str = os.path.join(DATA_DIR, "history.csv")
    MAX_HISTORY_RECORDS: int = 1000
    
    # Performance
    CACHE_EXPIRE: int = 3600  # 1 hour

settings = AppSettings()
