import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import os
import time
from configs.settings import settings
from configs.translations import TRANSLATIONS
from modules.detector import EnhancedEmotionDetector
from modules.utils import save_history, load_history

# ----------------- 初始化设置 -----------------
st.set_page_config(
    page_title="AI Emotion Detector Pro",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- 多语言支持 -----------------
lang = st.sidebar.selectbox(
    "🌐 Language",
    options=settings.AVAILABLE_LANGS,
    index=settings.AVAILABLE_LANGS.index(settings.DEFAULT_LANGUAGE)
)
T = TRANSLATIONS[lang]

# ----------------- 初始化检测器 -----------------
@st.cache_resource
def get_detector():
    detector = EnhancedEmotionDetector()
    return detector

detector = get_detector()

# ----------------- 主界面 -----------------
def main():
    st.title(f"😊 {T['title']}")
    st.caption(T['upload_guide'])
    
    # 用户认证
    username = st.text_input(f"👤 {T['username']}", value="user01")
    
    # 设置面板
    with st.expander("⚙️ " + T['settings'], expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            min_conf = st.slider(
                T['min_conf'],
                0.1, 1.0, float(detector.min_confidence), 0.05)
        with col2:
            detector_backend = st.selectbox(
                T['detector'],
                options=settings.DETECTOR_BACKENDS,
                index=settings.DETECTOR_BACKENDS.index(detector.detector_backend))
        
        detector.min_confidence = min_conf
        detector.detector_backend = detector_backend
    
    # 文件上传
    uploaded_file = st.file_uploader(T['upload_image'], type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        try:
            # 读取图片
            image = Image.open(uploaded_file)
            img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # 检测情绪
            with st.spinner(T['analyzing']):
                start_time = time.time()
                detections = detector.detect_emotions(img)
                process_time = time.time() - start_time
                
                if detections:
                    detected_img = detector.draw_detections(img, detections)
            
            # 显示结果
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader(T['analysis_results'])
                if detections:
                    st.success(f"✅ Detected {len(detections)} face(s) in {process_time:.2f}s")
                    
                    for i, det in enumerate(detections):
                        st.markdown(f"""
                        **Face {i+1}**
                        - **{T['detected_emotion']}**: `{det['emotion']}`
                        - **{T['confidence']}**: `{det['confidence']}%`
                        - Face Confidence: `{det['face_confidence']}%`
                        - Position: `({det['x']}, {det['y']})`
                        - Size: `{det['w']}x{det['h']}`
                        """)
                    
                    # 保存结果
                    save_history({
                        "username": username,
                        "emotion": detections[0]["emotion"],
                        "confidence": detections[0]["confidence"],
                        "face_confidence": detections[0]["face_confidence"],
                        "image_size": f"{image.width}x{image.height}",
                        "detector": detector.detector_backend,
                        "models": detector.models
                    })
                else:
                    st.warning(T['no_faces'])
            
            with col2:
                if detections:
                    tab1, tab2 = st.tabs([T['original'], T['processed']])
                    with tab1:
                        st.image(image, use_column_width=True)
                    with tab2:
                        st.image(detected_img, channels="BGR", use_column_width=True)
                else:
                    st.image(image, caption=T['original'], use_column_width=True)
        
        except Exception as e:
            st.error(f"{T['error_processing']}: {str(e)}")
    
    # 历史记录
    st.divider()
    st.subheader(f"📜 {T['history']}")
    
    if username:
        history_df = load_history(username)
        if not history_df.empty:
            st.dataframe(history_df)
            
            # 情绪分布图表
            st.plotly_chart(
                px.pie(history_df, names="emotion", title="Emotion Distribution"),
                use_container_width=True
            )
        else:
            st.info("No history records found")
    else:
        st.warning("Please enter username to view history")

if __name__ == "__main__":
    main()
