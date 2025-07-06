
```python
import os
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
from PIL import Image
from deepface import DeepFace
from utils.config import get_config
from utils.helpers import draw_detections, save_history

# Initialize configuration
config = get_config()

# Page setup
st.set_page_config(
    page_title=config["title"],
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Language selection
lang = st.sidebar.selectbox(
    config["language_selector"]["label"],
    options=list(config["translations"].keys()),
    index=0
)
T = config["translations"][lang]

def main():
    st.title(f"😊 {T['title']}")
    st.caption(T["upload_guide"])
    
    # User authentication
    username = st.text_input(f"👤 {T['username']}")
    
    # Main content
    if username:
        uploaded_file = st.file_uploader(
            T["upload_image"], 
            type=["jpg", "png", "jpeg"]
        )
        
        if uploaded_file:
            try:
                image = Image.open(uploaded_file)
                img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                
                # Emotion detection
                try:
                    results = DeepFace.analyze(
                        img_path=img,
                        actions=['emotion'],
                        enforce_detection=False,
                        detector_backend='opencv',
                        silent=True
                    )
                    
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
                    
                    # Display results
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.subheader(T["results_header"])
                        if emotions:
                            df = pd.DataFrame({
                                "Emotion": emotions,
                                "Confidence (%)": confidences
                            })
                            st.dataframe(df)
                            save_history(username, emotions[0], confidences[0])
                        else:
                            st.warning(T["no_faces"])
                    
                    with col2:
                        if emotions:
                            detected_img = draw_detections(img, emotions, faces, confidences, config)
                            st.image(detected_img, channels="BGR", use_column_width=True)
                        else:
                            st.image(image, caption=T["original_image"])
                
                except Exception as e:
                    st.error(f"{T['detection_error']}: {str(e)}")
            
            except Exception as e:
                st.error(f"{T['processing_error']}: {str(e)}")
    
    # History section
    st.divider()
    st.subheader(f"📜 {T['history']}")
    
    if os.path.exists("history/emotion_history.csv"):
        history = pd.read_csv("history/emotion_history.csv")
        
        if username:
            user_history = history[history["Username"] == username]
            
            if not user_history.empty:
                tab1, tab2 = st.tabs(["Data", "Charts"])
                
                with tab1:
                    st.dataframe(user_history)
                
                with tab2:
                    st.bar_chart(user_history["Emotion"].value_counts())
            else:
                st.info(T["no_user_history"])
        else:
            st.info(T["enter_username_history"])
    else:
        st.info(T["no_history"])

if __name__ == "__main__":
    main()
