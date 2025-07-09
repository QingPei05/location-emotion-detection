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

def save_history(username, emotions, confidences, location="Unknown"):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    records = []
    for i, (emo, conf) in enumerate(zip(emotions, confidences)):
        records.append([location, emo, conf, now])
    
    df = pd.DataFrame(records, columns=["Location", "Emotion", "Confidence", "timestamp"])
    try:
        if os.path.exists("history.csv"):
            prev = pd.read_csv("history.csv")
            df = pd.concat([prev, df])
        df.to_csv("history.csv", index=False)
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
    st.title("👁‍🗨 AI Emotion & Location Detector")
    st.subheader("👤 Sign In")
    username = st.text_input("Username", label_visibility="collapsed", placeholder="Username")
    password = st.text_input("Password", type="password", label_visibility="collapsed", placeholder="Password")
    
    # Buttons side by side
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Sign In"):
            if authenticate(username, password):
                st.session_state["logged_in"] = True
                st.session_state["username"] = username
                st.rerun()
            else:
                st.error("Invalid username or password")
    with col2:
        if st.button("Sign Up"):
            st.session_state["show_signup"] = True
            st.rerun()

def signup_page():
    st.title("👁‍🗨 AI Emotion & Location Detector")
    st.subheader("👤 Sign Up")
    username = st.text_input("Choose a username", label_visibility="collapsed", placeholder="Username")
    password = st.text_input("Choose a password", type="password", label_visibility="collapsed", placeholder="Password")
    confirm_password = st.text_input("Confirm password", type="password", label_visibility="collapsed", placeholder="Confirm Password")
    
    # Buttons side by side
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Register"):
            if password != confirm_password:
                st.error("Passwords don't match")
            elif register_user(username, password):
                st.success("Registration successful! Please sign in.")
                st.session_state["show_signup"] = False
                st.rerun()
            else:
                st.error("Username already exists or registration failed")
    with col2:
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
                        
                        # Add emotion totals
                        emotion_counts = {}
                        for emo in emotions:
                            emotion_counts[emo] = emotion_counts.get(emo, 0) + 1
                        total_text = "Total: " + ", ".join([f"{count} {emo}" for emo, count in emotion_counts.items()])
                        st.write(total_text)
                        
                        show_detection_guide()
                        save_history(username, emotions, confidences, "Unknown")
                    else:
                        st.warning("No faces were detected in the uploaded image.")
                with col2:
                    t1, t2 = st.tabs(["Original Image", "Processed Image"])
                    with t1:
                        st.image(image, use_column_width=True)
                    with t2:
                        st.image(detected_img, channels="BGR", use_column_width=True,
                                caption=f"Detected {len(detections)} {face_word}")
            except Exception as e:
                st.error(f"Error while processing the image: {e}")

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
                    # Display the history without username
                    edited_df = st.data_editor(
                        df[["Location", "Emotion", "Confidence", "timestamp"]],
                        key="history_editor"
                    )
                    
                    # Show details when row is selected
                    if "history_editor" in st.session_state:
                        selected_rows = st.session_state.history_editor["edited_rows"]
                        for idx, changes in selected_rows.items():
                            if changes:
                                with st.expander(f"Details for record {idx+1}"):
                                    st.write(f"Location: {df.iloc[idx]['Location']}")
                                    st.write(f"Emotion: {df.iloc[idx]['Emotion']}")
                                    st.write(f"Confidence: {df.iloc[idx]['Confidence']}%")
                                    st.write(f"Timestamp: {df.iloc[idx]['timestamp']}")
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
