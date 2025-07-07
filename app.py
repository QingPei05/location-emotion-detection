import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from datetime import datetime
import random
import os
import plotly.express as px
from emotion_utils.detector import EmotionDetector
import hashlib

# ----------------- User Authentication -----------------
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
    except:
        return False

def register_user(username, password):
    """Register new user"""
    try:
        # Check if username already exists
        if os.path.exists("users.csv"):
            users = pd.read_csv("users.csv")
            if username in users["username"].values:
                return False
        
        # Hash the password
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        
        # Create new user record
        new_user = pd.DataFrame([[username, hashed_password]], 
                              columns=["username", "password"])
        
        # Append to existing users or create new file
        if os.path.exists("users.csv"):
            new_user.to_csv("users.csv", mode='a', header=False, index=False)
        else:
            new_user.to_csv("users.csv", index=False)
            
        return True
    except Exception as e:
        print(f"Registration error: {e}")
        return False

# ----------------- App Configuration -----------------
st.set_page_config(
    page_title="AI Emotion & Location Detector",
    page_icon="👁‍🗨",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def get_detector():
    return EmotionDetector()

detector = get_detector()

def save_history(username, image_id, emotions, confidences, location="Unknown"):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Count each emotion type
    emotion_counts = {}
    for emo in emotions:
        emotion_counts[emo] = emotion_counts.get(emo, 0) + 1
    
    # Format emotions string (e.g. "2 happy, 1 sad")
    emotions_str = ", ".join([f"{count} {emotion}" for emotion, count in emotion_counts.items()])
    
    # Save image with unique ID
    image_path = f"history_images/{image_id}.jpg"
    os.makedirs("history_images", exist_ok=True)
    cv2.imwrite(image_path, cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
    
    # Create record
    record = {
        "location": location,
        "emotions": emotions_str,
        "timestamp": now,
        "image_id": image_id,
        "all_emotions": ",".join(emotions),
        "all_confidences": ",".join(map(str, confidences))
    }
    
    # Save to CSV
    try:
        if os.path.exists("history.csv"):
            history_df = pd.read_csv("history.csv")
            history_df = pd.concat([history_df, pd.DataFrame([record])])
        else:
            history_df = pd.DataFrame([record])
        
        history_df.to_csv("history.csv", index=False)
    except Exception as e:
        st.error(f"Failed to save history: {e}")

def show_detection_guide():
    with st.expander("ℹ️ How Emotion Detection Works", expanded=False):
        st.markdown("""
        *Detection Logic Explained:*
        - 😊 Happy: Smile present, cheeks raised
        - 😠 Angry: Eyebrows lowered, eyes wide open
        - 😐 Neutral: No strong facial movements
        - 😢 Sad: Eyebrows raised, lip corners down
        - 😲 Surprise: Eyebrows raised, mouth open
        - 😨 Fear: Eyes tense, lips stretched
        - 🤢 Disgust: Nose wrinkled, upper lip raised

        *Tips for Better Results:*
        - Use clear, front-facing images
        - Ensure good lighting
        - Avoid obstructed faces
        """)

def sidebar_design(username):
    """Design the sidebar with user info and navigation"""
    if username:  # Only show if username exists
        st.sidebar.success(f"👤 Logged in as: {username}")
    st.sidebar.markdown("---")
    st.sidebar.markdown("## Quick Navigation")
    st.sidebar.markdown("- Upload and detect emotions")
    st.sidebar.markdown("- View and filter upload history")
    st.sidebar.markdown("- Visualize your emotion distribution")
    st.sidebar.divider()
    st.sidebar.info("Enhance your experience by ensuring clear, well-lit facial images.")
    
    # Add logout button
    if st.sidebar.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.rerun()

# ----------------- Login/Signup Pages -----------------
def login_page():
    st.title("👤 Sign In")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Sign In"):
        if authenticate(username, password):
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
            st.rerun()
        else:
            st.error("Invalid username or password")
    
    if st.button("Sign Up"):
        st.session_state["show_signup"] = True
        st.rerun()

def signup_page():
    st.title("👤 Sign Up")
    username = st.text_input("Choose a username")
    password = st.text_input("Choose a password", type="password")
    confirm_password = st.text_input("Confirm password", type="password")
    
    if st.button("Register"):
        if password != confirm_password:
            st.error("Passwords don't match")
        elif register_user(username, password):
            st.success("Registration successful! Please sign in.")
            st.session_state["show_signup"] = False
            st.rerun()
        else:
            st.error("Username already exists or registration failed")
    
    if st.button("Back to Sign In"):
        st.session_state["show_signup"] = False
        st.rerun()

# ----------------- Main App -----------------
def main_app():
    username = st.session_state.get("username", "")
    sidebar_design(username)
    
    st.title("👁‍🗨 AI Emotion & Location Detector")
    st.caption("Upload a photo to detect facial emotions and estimate location.")
    
    tabs = st.tabs(["🏠 Home", "🗺️ Location Map", "📜 Upload History", "📊 Emotion Analysis Chart"])

    with tabs[0]:
    uploaded_file = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "png"])
    if uploaded_file:
        try:
            image = Image.open(uploaded_file)
            img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            detections = detector.detect_emotions(img)
            detected_img = detector.draw_detections(img, detections)
            
            # Generate unique ID for this image
            image_id = hashlib.md5(uploaded_file.getvalue()).hexdigest()
            
            # Detect location from image (implementation below)
            location = detect_location(img)  
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.subheader("🔍 Detection Results")
                if detections:
                    emotions = [d["emotion"] for d in detections]
                    confidences = [d["confidence"] for d in detections]
                    
                    # Correct pluralization
                    face_word = "face" if len(detections) == 1 else "faces"
                    st.success(f"🎭 {len(detections)} {face_word} detected")
                    
                    for i, (emo, conf) in enumerate(zip(emotions, confidences)):
                        st.write(f"- Face {i + 1}: {emo} ({conf}%)")
                    
                    # Display detected location
                    st.write(f"📍 Location: {location if location else 'Unknown'}")
                    
                    show_detection_guide()
                    save_history(username, image_id, emotions, confidences, location if location else "Unknown")

            with col2:
                t1, t2 = st.tabs(["Original Image", "Processed Image"])
                with t1:
                    st.image(image, use_column_width=True)
                with t2:
                    st.image(detected_img, channels="BGR", use_column_width=True,
                            caption=f"Detected {len(detections)} {face_word}")
        except Exception as e:
            st.error(f"Error while processing the image: {e}")

# Add this function to your app.py (place it with your other utility functions)
def detect_location(img):
    """Detect location from image using EXIF data or other methods"""
    try:
        # Method 1: Check for EXIF GPS data
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        exif_data = pil_img._getexif()
        
        if exif_data:
            from PIL.ExifTags import TAGS, GPSTAGS
            gps_info = {}
            for tag, value in exif_data.items():
                decoded = TAGS.get(tag, tag)
                if decoded == "GPSInfo":
                    for t in value:
                        sub_decoded = GPSTAGS.get(t, t)
                        gps_info[sub_decoded] = value[t]
            
            if 'GPSLatitude' in gps_info and 'GPSLongitude' in gps_info:
                def convert_to_degrees(value):
                    return float(value[0]) + float(value[1])/60 + float(value[2])/3600
                
                lat = convert_to_degrees(gps_info['GPSLatitude'])
                lon = convert_to_degrees(gps_info['GPSLongitude'])
                
                if gps_info.get('GPSLatitudeRef') == 'S':
                    lat = -lat
                if gps_info.get('GPSLongitudeRef') == 'W':
                    lon = -lon
                
                from geopy.geocoders import Nominatim
                geolocator = Nominatim(user_agent="emotion_detector")
                location = geolocator.reverse(f"{lat}, {lon}", exactly_one=True)
                return location.address if location else None
        
        # Method 2: If no EXIF data, use a placeholder (you can implement other methods here)
        return None
        
    except Exception as e:
        print(f"Location detection error: {e}")
        return None

    with tabs[1]:
        st.subheader("🗺️ Random Location Sample (Demo)")
        st.map(pd.DataFrame({
            'lat': [3.139 + random.uniform(-0.01, 0.01)],
            'lon': [101.6869 + random.uniform(-0.01, 0.01)]
        }))
        st.caption("Note: This location map is a demo preview and not actual detected GPS data.")

    with tabs[2]:
    st.subheader("📜 Upload History")
    try:
        if os.path.exists("history.csv"):
            df = pd.read_csv("history.csv")
            if df.empty:
                st.info("No upload records found.")
            else:
                # Display simplified history table
                st.dataframe(df[["location", "emotions", "timestamp"]], 
                            use_container_width=True)
                
                # Add selection functionality
                selected_index = st.selectbox(
                    "Select a record to view details:",
                    range(len(df)),
                    format_func=lambda x: f"Record {x+1} - {df.iloc[x]['timestamp']}"
                )
                
                if selected_index is not None:
                    record = df.iloc[selected_index]
                    with st.expander(f"📄 Details for {record['timestamp']}", expanded=True):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Show original image
                            image_path = f"history_images/{record['image_id']}.jpg"
                            if os.path.exists(image_path):
                                st.image(image_path, caption="Original Image", use_column_width=True)
                            else:
                                st.warning("Original image not found")
                            
                        with col2:
                            # Show emotion analysis
                            st.write(f"**Location:** {record['location']}")
                            st.write(f"**Timestamp:** {record['timestamp']}")
                            
                            # Create pie chart for this image's emotions
                            emotions = record['all_emotions'].split(',')
                            confidences = list(map(float, record['all_confidences'].split(',')))
                            
                            fig = px.pie(
                                names=emotions,
                                values=confidences,
                                title="Emotion Distribution for This Image"
                            )
                            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No history file found.")
    except Exception as e:
        st.warning(f"Error loading history records: {e}")

    with tabs[3]:
        st.subheader("📊 Emotion Analysis Chart")
        try:
            if os.path.exists("history.csv"):
                df = pd.read_csv("history.csv")
                if df.empty:
                    st.info("No emotion records found.")
                else:
                    fig = px.pie(df, names="Emotion", title="Emotion Distribution")
                    st.plotly_chart(fig)
                    st.caption("Chart shows distribution of all detected emotions")
            else:
                st.info("No history file found.")
        except Exception as e:
            st.error(f"Error generating chart: {e}")

# ----------------- Run App -----------------
if __name__ == "__main__":
    # Initialize session state variables
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "show_signup" not in st.session_state:
        st.session_state.show_signup = False
    if "username" not in st.session_state:
        st.session_state.username = ""

    # Authentication flow
    if not st.session_state.logged_in:
        if st.session_state.show_signup:
            signup_page()
        else:
            login_page()
    else:
        try:
            main_app()
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.stop()
