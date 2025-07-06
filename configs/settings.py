class Settings:
    # 检测配置
    MIN_CONFIDENCE = 0.65
    DETECTOR_BACKEND = "mtcnn"  # 可选: opencv, ssd, dlib, mtcnn, retinaface
    MODELS = ["VGG-Face", "Facenet", "OpenFace"]
    
    # 界面配置
    DEFAULT_LANGUAGE = "English"
    AVAILABLE_LANGS = ["English", "中文", "Malay"]
    
    # 历史记录
    HISTORY_FILE = "data/history.csv"
    MAX_HISTORY = 1000

settings = Settings()
