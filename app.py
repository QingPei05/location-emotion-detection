import os
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from datetime import datetime
import plotly.express as px
from emotion_utils.detector import EmotionDetector
import hashlib
import tempfile
from location_utils.extract_gps import extract_gps, convert_gps
from location_utils.geocoder import get_address_from_coords
from location_utils.landmark import LANDMARK_KEYWORDS, detect_landmark, query_landmark_coords
import concurrent.futures

# ----------------- Configuration -----------------
st.set_page_config(
    page_title="Perspēct",
    page_icon="👁‍🗨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- Cached Resources -----------------
@st.cache_resource
def load_models():
    """Load and cache all ML models"""
    return {
        'emotion': EmotionDetector(),
        # Other models can be added here
    }

@st.cache_data(ttl=3600, show_spinner="Processing image...")
def process_uploaded_image(uploaded_file):
    """Cache processed image data to avoid redundant computations"""
    try:
        image = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(image)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        return {
            'pil_image': image,
            'img_np': img_np,
            'img_bgr': img_bgr,
            'file_hash': hashlib.md5(uploaded_file.getvalue()).hexdigest()
        }
    except Exception as e:
        st.error(f"Image processing error: {str(e)}")
        return None

# ----------------- Authentication -----------------
def authenticate(username, password):
    """Check if username and password match"""
    try:
        if os.path.exists("users.csv"):
            users = pd.read_csv("users.csv")
            user_record = users[users["username"] == username]
            if not user_record.empty:
                hashed_password = hashlib.sha256(password.encode()).hexdigest()
                return user_record["password"].values[0] == hashed_password
        return False
    except Exception:
        return False

def register_user(username, password):
    """Register new user"""
    try:
        if os.path.exists("users.csv"):
            users = pd.read_csv("users.csv")
            if username in users["username"].values:
                return False
        
        new_user = pd.DataFrame([[username, hashlib.sha256(password.encode()).hexdigest()]], 
                              columns=["username", "password"])
        
        new_user.to_csv("users.csv", mode='a' if os.path.exists("users.csv") else 'w', header=not os.path.exists("users.csv"), index=False)
        return True
    except Exception as e:
        st.error(f"Registration error: {e}")
        return False

# ----------------- UI Components -----------------
def gradient_card(subtitle):
    st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #fef9ff, #e7e7f9);
            border-radius: 20px;
            padding: 2rem;
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.06);
            text-align: center;
            border: 1px solid #ddd;
            margin-bottom: 2rem;
        ">
            <h1 style="color: #5a189a; font-size: 2.8rem;">👁‍🗨 Perspēct</h1>
            {f'<p style="color: #333; font-size: 1.2rem;">{subtitle}</p>' if subtitle else ''}
        </div>
    """, unsafe_allow_html=True)

# ----------------- Processing Functions -----------------
def process_emotion(image_data, detector):
    """Process emotions using cached image data"""
    try:
        detections = detector.detect_emotions(image_data['img_bgr'])
        detected_img = detector.draw_detections(image_data['img_bgr'], detections)
        return detections, detected_img
    except Exception as e:
        st.error(f"Emotion detection failed: {str(e)}")
        return [], None

def process_location(temp_path):
    """Process location data"""
    try:
        if gps_info := extract_gps(temp_path):
            if coords := convert_gps(gps_info):
                return coords, "GPS Metadata"
        
        if landmark := detect_landmark(temp_path, threshold=0.15, top_k=3):
            if coords_loc := query_landmark_coords(landmark)[0]:
                return coords_loc, f"Landmark (CLIP)"
        return None, ""
    except Exception as e:
        st.error(f"Location detection failed: {str(e)}")
        return None, ""

# ----------------- Main App -----------------
def main_app():
    models = load_models()
    username = st.session_state.get("username", "")
    
    # Initialize session state
    for key in ['coords_result', 'location_method', 'landmark', 'show_history']:
        st.session_state.setdefault(key, None)

    gradient_card("Upload a photo to detect facial emotions and estimate location")
    
    if st.session_state.show_history:
        show_user_history(username)
    else:
        tabs = st.tabs(["🏠 Home", "🗺️ Location Map"])
        
        with tabs[0]:
            if uploaded_file := st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "png"]):
                image_data = process_uploaded_image(uploaded_file)
                
                if image_data:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        temp_path = tmp_file.name
                    
                    try:
                        with st.spinner('Analyzing image...'):
                            with concurrent.futures.ThreadPoolExecutor() as executor:
                                emotion_future = executor.submit(process_emotion, image_data, models['emotion'])
                                location_future = executor.submit(process_location, temp_path)
                                
                                detections, detected_img = emotion_future.result()
                                coords, method = location_future.result()
                                
                            # Process results
                            st.session_state.update({
                                'coords_result': coords,
                                'location_method': method,
                                'landmark': detect_landmark(temp_path) if coords else None
                            })
                            
                            display_results(image_data, detections, detected_img)
                            
                    finally:
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
        
        with tabs[1]:
            display_location_map()

# ----------------- Result Display -----------------
def display_results(image_data, detections, detected_img):
    """Display processed results"""
    if not detections:
        st.warning("No faces detected")
        return
    
    emotions = [d["emotion"] for d in detections]
    confidences = [d["confidence"] for d in detections]
    face_word = "face" if len(detections) == 1 else "faces"
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("🔍 Detection Results")
        st.success(f"🎭 {len(detections)} {face_word} detected")
        
        with st.expander("View details"):
            for i, (emo, conf) in enumerate(zip(emotions, confidences)):
                st.write(f"**Face {i+1}**: {emo.title()} ({conf:.1f}%)")
            
            st.write("**Summary**:", ", ".join(
                f"{emotions.count(e)} {e}" for e in set(emotions)
            ))
        
        location = get_location_string()
        st.success(f"📍 {location}")
        
    with col2:
        tab1, tab2 = st.tabs(["Original", "Processed"])
        with tab1:
            st.image(image_data['pil_image'], use_column_width=True)
        with tab2:
            st.image(detected_img, channels="BGR", use_column_width=True,
                   caption=f"Detected {len(detections)} {face_word}")

def get_location_string():
    """Generate location description string"""
    if not st.session_state.coords_result:
        return "Location unknown"
    
    coords = st.session_state.coords_result
    if address := get_address_from_coords(coords):
        if address not in ("Unknown location", "Geocoding service unavailable"):
            return address
    
    if landmark := st.session_state.landmark:
        if info := LANDMARK_KEYWORDS.get(landmark):
            return f"{info[0]}, {info[1]}"
        return f"{landmark.title()} ({coords[0]:.4f}, {coords[1]:.4f})"
    return f"GPS: {coords[0]:.4f}, {coords[1]:.4f}"

def display_location_map():
    """Display location map if available"""
    st.subheader("🗺️ Location Map")
    
    if coords := st.session_state.coords_result:
        st.write(f"**Method**: {st.session_state.location_method}")
        st.write(f"**Location**: {get_location_string()}")
        st.map(pd.DataFrame({"lat": [coords[0]], "lon": [coords[1]]}))
    else:
        st.warning("No location data available")

# ----------------- Run App -----------------
if __name__ == "__main__":
    # Initialize session state
    for key in ['logged_in', 'show_signup', 'username']:
        st.session_state.setdefault(key, False if key != 'username' else "")
    
    if not st.session_state.logged_in:
        if st.session_state.show_signup:
            signup_page()
        else:
            login_page()
    else:
        try:
            main_app()
        except Exception as e:
            st.error(f"Application error: {str(e)}")
