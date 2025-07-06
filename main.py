import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from datetime import datetime
import os
from deepface import DeepFace

# ----------------- 初始化设置 -----------------
st.set_page_config(
    page_title="AI Emotion Detector (Enhanced)",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- 多语言支持 -----------------
LANGUAGES = ["中文", "English", "Malay"]
lang = st.sidebar.selectbox("🌐 Language", LANGUAGES)

TRANSLATIONS = {
    "中文": {
        "title": "AI情绪检测系统 (增强版)",
        "upload_guide": "上传照片分析面部表情",
        "username": "用户名",
        "enter_username": "输入用户名",
        "upload_image": "上传图片 (JPG/PNG)",
        "detected_emotion": "检测到的情绪",
        "confidence": "置信度",
        "no_faces": "未检测到人脸",
        "history": "历史记录"
    },
    "English": {
        "title": "AI Emotion Detector (Enhanced)",
        "upload_guide": "Upload photo to analyze facial expressions",
        "username": "Username",
        "enter_username": "Enter username",
        "upload_image": "Upload image (JPG/PNG)",
        "detected_emotion": "Detected Emotion",
        "confidence": "Confidence",
        "no_faces": "No faces detected",
        "history": "History"
    },
    "Malay": {
        "title": "Pengesan Emosi AI (Tingkat Baik)",
        "upload_guide": "Muat naik foto untuk analisis ekspresi muka",
        "username": "Nama pengguna",
        "enter_username": "Masukkan nama pengguna",
        "upload_image": "Muat naik imej (JPG/PNG)",
        "detected_emotion": "Emosi yang Dikesan",
        "confidence": "Keyakinan",
        "no_faces": "Tiada muka dikesan",
        "history": "Sejarah"
    }
}
T = TRANSLATIONS[lang]

# ----------------- 情绪检测核心功能 -----------------
def detect_emotion_deepface(img):
    try:
        # Convert to RGB (DeepFace expects RGB)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Analyze with DeepFace
        results = DeepFace.analyze(
            img_path=img_rgb,
            actions=['emotion'],
            enforce_detection=False,
            detector_backend='opencv',
            silent=True
        )
        
        # Process results
        emotions = []
        faces = []
        confidences = []
        
        for result in results:
            emotion = result['dominant_emotion']
            confidence = result['emotion'][emotion]
            region = result['region']
            
            emotions.append(emotion)
            confidences.append(round(confidence, 2))
            faces.append((region['x'], region['y'], region['w'], region['h']))
            
        return emotions, faces, confidences
    except Exception as e:
        st.error(f"Detection error: {str(e)}")
        return [], [], []

# ----------------- 可视化函数 -----------------
def draw_detections(img, emotions, faces, confidences):
    """Draw detection boxes with labels and confidence"""
    output_img = img.copy()
    
    # 颜色映射 (新增了更多情绪颜色)
    color_map = {
        "happy": (0, 255, 0),      # 绿色
        "neutral": (255, 255, 0),  # 黄色
        "sad": (0, 0, 255),        # 红色
        "angry": (0, 165, 255),   # 橙色
        "fear": (128, 0, 128),     # 紫色
        "surprise": (255, 0, 255), # 粉色
        "disgust": (0, 128, 0)     # 深绿
    }
    
    for i, ((x,y,w,h), emotion, confidence) in enumerate(zip(faces, emotions, confidences)):
        color = color_map.get(emotion, (255, 255, 255))
        
        # 绘制人脸矩形
        cv2.rectangle(output_img, (x,y), (x+w,y+h), color, 3)
        
        # 添加情绪标签和置信度
        label = f"{emotion} ({confidence}%)"
        cv2.putText(output_img, 
                   label, 
                   (x+5, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   0.8, 
                   color, 
                   2)
    
    return output_img

# ----------------- 历史记录功能 -----------------
def save_history(username, emotion, confidence):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_entry = pd.DataFrame([[username, emotion, confidence, now]], 
                            columns=["Username","Emotion","Confidence","Timestamp"])
    
    try:
        if os.path.exists("history.csv"):
            history = pd.read_csv("history.csv")
            history = pd.concat([history, new_entry])
        else:
            history = new_entry
            
        history.to_csv("history.csv", index=False)
    except Exception as e:
        st.error(f"Failed to save history: {str(e)}")

# ----------------- 主界面 -----------------
def main():
    st.title(f"😊 {T['title']}")
    st.caption(T['upload_guide'])
    
    # 用户认证
    username = st.text_input(f"👤 {T['enter_username']}")
    
    # 主内容区
    if username:
        uploaded_file = st.file_uploader(T['upload_image'], type=["jpg","png"])
        
        if uploaded_file:
            try:
                # 读取图片
                image = Image.open(uploaded_file)
                img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                
                # 检测情绪
                emotions, faces, confidences = detect_emotion_deepface(img)
                
                # 显示结果
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.subheader("🔍 Detection Results")
                    if emotions:
                        # 显示情绪统计
                        emotion_df = pd.DataFrame({
                            "Emotion": emotions,
                            "Confidence": confidences
                        })
                        
                        st.dataframe(emotion_df.style.highlight_max(axis=0))
                        
                        # 保存第一条检测结果到历史
                        save_history(username, emotions[0], confidences[0])
                    else:
                        st.warning(T['no_faces'])
                
                with col2:
                    if emotions:
                        # 绘制检测结果
                        detected_img = draw_detections(img, emotions, faces, confidences)
                        st.image(detected_img, channels="BGR", 
                                 caption=f"Detected {len(faces)} face(s)")
                    else:
                        st.image(image, caption="Original Image")
            
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
    
    # 历史记录标签页
    st.divider()
    st.subheader(f"📜 {T['history']}")
    
    if os.path.exists("history.csv"):
        history = pd.read_csv("history.csv")
        
        if username:
            user_history = history[history["Username"] == username]
            st.dataframe(user_history)
            
            # 情绪分布图表
            if not user_history.empty:
                st.bar_chart(user_history["Emotion"].value_counts())
        else:
            st.info("Please enter username to view history")
    else:
        st.info("No history available yet")

if __name__ == "__main__":
    main()
