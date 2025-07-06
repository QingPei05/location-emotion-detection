def get_config():
    return {
        "title": "AI Emotion Detector",
        "language_selector": {
            "label": "🌐 Select Language"
        },
        "color_map": {
            "happy": (0, 255, 0),      # Green
            "neutral": (255, 255, 0),   # Yellow
            "sad": (0, 0, 255),         # Red
            "angry": (0, 165, 255),     # Orange
            "fear": (128, 0, 128),      # Purple
            "surprise": (255, 0, 255),  # Pink
            "disgust": (0, 128, 0)      # Dark Green
        },
        "translations": {
            "English": {
                "title": "AI Emotion Detector",
                "upload_guide": "Upload photo to analyze facial expressions",
                "username": "Username",
                "upload_image": "Upload image (JPG/PNG)",
                "results_header": "🔍 Detection Results",
                "no_faces": "No faces detected",
                "original_image": "Original Image",
                "detection_error": "Detection error",
                "processing_error": "Image processing error",
                "history": "Detection History",
                "no_user_history": "No history found for this user",
                "enter_username_history": "Please enter username to view history",
                "no_history": "No detection history available"
            },
            "中文": {
                "title": "AI情绪检测系统",
                "upload_guide": "上传照片分析面部表情",
                "username": "用户名",
                "upload_image": "上传图片 (JPG/PNG)",
                "results_header": "🔍 检测结果",
                "no_faces": "未检测到人脸",
                "original_image": "原始图片",
                "detection_error": "检测错误",
                "processing_error": "图片处理错误",
                "history": "检测历史",
                "no_user_history": "未找到该用户的历史记录",
                "enter_username_history": "请输入用户名查看历史",
                "no_history": "暂无检测历史"
            }
        }
    }
