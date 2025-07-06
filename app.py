# ai_emotion_location_app.py (debugged version)

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from datetime import datetime
import random
import logging
import os
from io import BytesIO
import plotly.express as px

# ----------------- 初始化设置 -----------------
st.set_page_config(
    page_title="AI Emotion & Location Detector",
    page_icon="👁‍🗨",
    layout="wide",
    initial_sidebar_state="expanded"
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------- 多语言支持 -----------------
LANGUAGES = ["中文", "English", "Malay"]
lang = st.sidebar.selectbox("🌐 Select Language / 选择语言 / Pilih Bahasa", LANGUAGES)

TRANSLATIONS = {
    "中文": {
        "title": "AI情绪与位置检测系统",
        "upload_guide": "上传照片分析面部表情并推测位置",
        "username": "用户名",
        "user_auth": "用户认证",
        "enter_username": "输入用户名",
        "welcome": "欢迎",
        "upload_image": "上传图片 (JPG/PNG)",
        "analysis_results": "分析结果",
        "detected_emotion": "检测到的情绪",
        "estimated_location": "估计位置",
        "download_results": "下载结果",
        "original_image": "原始图片",
        "processed_image": "处理后的图片",
        "no_faces": "未检测到人脸",
        "error_processing": "图片处理错误",
        "debug_info": "调试信息",
        "input_username_continue": "请输入用户名继续",
        "nav_home": "主页",
        "nav_location_map": "位置地图",
        "nav_history": "历史记录",
        "nav_emotion_chart": "情绪分析图表",  
        "upload_history": "上传历史",
        "no_history": "暂无历史记录",
        "filter_user": "按用户名筛选",
        "records_shown": "条记录已显示",
        "no_record_found": "未找到记录",
        "enter_username_history": "请输入用户名查看历史"
    },
    "English": {
        "title": "AI Emotion & Location Detector",
        "upload_guide": "Upload a photo to analyze facial expressions and estimate location",
        "username": "Username",
        "user_auth": "User Authentication",
        "enter_username": "Enter your username",
        "welcome": "Welcome",
        "upload_image": "Upload an image (JPG/PNG)",
        "analysis_results": "Analysis Results",
        "detected_emotion": "Detected Emotion",
        "estimated_location": "Estimated Location",
        "download_results": "Download Results",
        "original_image": "Original Image",
        "processed_image": "Processed Image",
        "no_faces": "No faces detected",
        "error_processing": "Error processing image",
        "debug_info": "Debug Info",
        "input_username_continue": "Please enter a username to continue",
        "nav_home": "Home",
        "nav_location_map": "Location Map",
        "nav_history": "History",
        "nav_emotion_chart": "Emotion Chart",  
        "upload_history": "Upload History",
        "no_history": "No history available",
        "filter_user": "Filter by username",
        "records_shown": "records shown",
        "no_record_found": "No records found",
        "enter_username_history": "Please enter username to view history"
    },
    "Malay": {
        "title": "Sistem Pengesanan Emosi & Lokasi AI",
        "upload_guide": "Muat naik foto untuk analisis ekspresi muka dan anggaran lokasi",
        "username": "Nama pengguna",
        "user_auth": "Pengesahan Pengguna",
        "enter_username": "Masukkan nama pengguna",
        "welcome": "Selamat datang",
        "upload_image": "Muat naik imej (JPG/PNG)",
        "analysis_results": "Keputusan Analisis",
        "detected_emotion": "Emosi yang Dikesan",
        "estimated_location": "Lokasi Dianggarkan",
        "download_results": "Muat Turun Keputusan",
        "original_image": "Imej Asal",
        "processed_image": "Imej Diproses",
        "no_faces": "Tiada muka dikesan",
        "error_processing": "Ralat memproses imej",
        "debug_info": "Maklumat Debug",
        "input_username_continue": "Masukkan nama pengguna untuk meneruskan",
        "nav_home": "Halaman Utama",
        "nav_location_map": "Peta Lokasi",
        "nav_history": "Sejarah",
        "nav_emotion_chart": "Carta Emosi",  
        "upload_history": "Sejarah Muat Naik",
        "no_history": "Tiada sejarah tersedia",
        "filter_user": "Tapis mengikut nama pengguna",
        "records_shown": "rekod dipaparkan",
        "no_record_found": "Tiada rekod dijumpai",
        "enter_username_history": "Sila masukkan nama pengguna untuk melihat sejarah"
    }
}
T = TRANSLATIONS[lang]

# ----------------- 加载模型 -----------------
@st.cache_resource
def load_models():
    face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    smile = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
    return face, eye, smile

face_cascade, eye_cascade, smile_cascade = load_models()

# ----------------- 核心功能 -----------------
def detect_emotion(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    emotions = []
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20)
        eyes = eye_cascade.detectMultiScale(roi_gray)
        emotion = "neutral"
        if len(eyes) >= 2:
            eye_sizes = [eh for (_,ey,_,eh) in eyes[:2]]
            avg_eye_size = np.mean(eye_sizes)
            eye_centers = [ey + eh/2 for (_,ey,_,eh) in eyes[:2]]
            avg_eye_height = np.mean(eye_centers)
            if avg_eye_size > h/5 and avg_eye_height < h/2.5:
                emotion = "angry"
            elif avg_eye_height < h/3:
                emotion = "sad"
        if len(smiles) > 0:
            emotion = "happy"
        emotions.append(emotion)
    return emotions, faces

def draw_detections(img, emotions, faces):
    """Draw detection boxes with English labels"""
    output_img = img.copy()
    
    # Color mapping
    color_map = {
        "happy": (0, 255, 0),     # green
        "neutral": (255, 255, 0), # yellow
        "sad": (0, 0, 255),       # red
        "angry": (0, 165, 255)    # orange
    }
    
    for i, ((x,y,w,h), emotion) in enumerate(zip(faces, emotions)):
        color = color_map.get(emotion, (255, 255, 255))
        
        # Draw face rectangle
        cv2.rectangle(output_img, (x,y), (x+w,y+h), color, 3)
        
        # Add emotion label
        cv2.putText(output_img, 
                   emotion.upper(), 
                   (x+5, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   0.8, 
                   color, 
                   2)
    
    return output_img

def show_detection_guide(show_full_guide=True):
    """Show detection guide in expandable section"""
    with st.expander("ℹ️ How Emotion Detection Works", expanded=False):
        if show_full_guide:
            st.markdown("""
            *Detection Logic Explained:*
            
            - 😊 *Happy*: Detected when smile is present
            - 😠 *Angry*: Detected when eyes are wide open and positioned in upper face
            - 😐 *Neutral*: Default state when no strong indicators found
            - 😢 *Sad*: Detected when eyes are positioned higher than normal
            
            *Tips for Better Results:*
            - Use clear, front-facing images
            - Ensure good lighting
            - Avoid obstructed faces
            """)
        else:
            st.markdown("""
            *Tips for Better Results:*
            - Use clear, front-facing images
            - Ensure good lighting
            - Avoid obstructed faces
            """)

def save_history(username, emotion, location):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df = pd.DataFrame([[username, emotion, location, now]], columns=["Username","Emotion","Location","timestamp"])
    try:
        if os.path.exists("history.csv"):
            prev = pd.read_csv("history.csv")
            df = pd.concat([prev, df])
        df.to_csv("history.csv", index=False)
    except Exception as e:
        logger.error(f"Save failed: {e}")

# ----------------- 主程序 -----------------
def main():
    st.title(f"👁‍🗨 {T['title']}")
    st.caption(T['upload_guide'])
    tabs = st.tabs([f"🏠 {T['nav_home']}", f"🗺️ {T['nav_location_map']}", f"📜 {T['nav_history']}", f"📊 {T['nav_emotion_chart']}"])

    # 首页
    with tabs[0]:
        username = st.text_input(f"👤 {T['enter_username']}")
        if username:
            st.sidebar.success(f"👤 {T['welcome']} {username}")
            uploaded_file = st.file_uploader(T['upload_image'], type=["jpg","png"])
            if uploaded_file:
                try:
                    image = Image.open(uploaded_file)
                    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    emotions, faces = detect_emotion(img)
                    detected_img = draw_detections(img, emotions, faces)

                    col1, col2 = st.columns([1,2])
                    with col1:
                        st.subheader("🔍 Detection Results")
                        if emotions:
                            emo_count = {e: emotions.count(e) for e in set(emotions)}
                            st.success(f"🎭 {len(faces)} face(s): {', '.join(f'{v} {k}' for k,v in emo_count.items())}")
                            show_detection_guide(True)
                            save_history(username, emotions[0], "Unknown")
                        else:
                            st.warning(T["no_faces"])
                            show_detection_guide(False)
                    with col2:
                        t1, t2 = st.tabs([T["original_image"], T["processed_image"]])
                        with t1: st.image(image, use_container_width=True)
                        with t2: st.image(detected_img, channels="BGR", use_container_width=True, 
                                          caption=f"Detected {len(faces)} faces")
                except Exception as e:
                    st.error(f"{T['error_processing']}: {e}")

    # 地图页
    with tabs[1]:
        st.map(pd.DataFrame({'lat':[3.139+random.uniform(-0.01,0.01)], 'lon':[101.6869+random.uniform(-0.01,0.01)]}))

    # 历史记录
    with tabs[2]:
        st.header(f"📜 {T['upload_history']}")
        if username:
            try:
                df = pd.read_csv("history.csv")
                if df.empty:
                    st.info(T['no_history'])
                else:
                    keyword = st.text_input(T['filter_user']).strip()
                    df_filtered = df[df["Username"].str.contains(keyword, case=False)] if keyword else df
                    st.dataframe(df_filtered)
                    st.caption(f"{len(df_filtered)} {T['records_shown']}")
            except:
                st.info(T['no_record_found'])
        else:
            st.warning(T['enter_username_history'])

    # 图表
    with tabs[3]:
        st.subheader(f"📊 {T['nav_emotion_chart']}")
        try:
            if os.path.exists("history.csv"):
                df = pd.read_csv("history.csv")
                if not df.empty:
                    fig = px.pie(df, names="Emotion", title=T["nav_emotion_chart"])
                    st.plotly_chart(fig)
                else:
                    st.warning(T["no_history"])
            else:
                st.warning(T["no_record_found"])
        except Exception as e:
            st.error(f"Chart error: {e}")

if __name__ == "__main__":
    main()
