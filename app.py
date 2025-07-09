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
    page_title="Perspēct",
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
        records.append([username, location, emo, conf, now])
    
    df = pd.DataFrame(records, columns=["username", "Location", "Emotion", "Confidence", "timestamp"])
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
    
    # Make all sidebar sections consistent in length
    st.sidebar.markdown("---")
    st.sidebar.markdown("## Quick Navigation")
    st.sidebar.markdown("- Upload and detect emotions")
    st.sidebar.markdown("- View location map")
    st.sidebar.divider()

    st.sidebar.info("Enhance your experience by ensuring clear, well-lit facial images.")
    st.sidebar.divider()
     # History button moved here
    if username:
        if st.sidebar.button("📜 History", key="history_button"):
            st.session_state.show_history = not st.session_state.get('show_history', False)
    
    # Add logout button
    if st.sidebar.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.show_history = False
        st.rerun()

def show_user_history(username):
    """Show user-specific history in main content area"""
    # Add back button in top right
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("📜 Your History")
    with col2:
        if st.button("⬅ Back to Main"):
            st.session_state.show_history = False
            st.rerun()
    
    try:
        if os.path.exists("history.csv"):
            df = pd.read_csv("history.csv")
            if not df.empty:
                # Check if username column exists, if not create empty dataframe
                if 'username' not in df.columns:
                    df['username'] = ""
                
                # Filter for current user only
                user_df = df[df["username"] == username]
                
                if not user_df.empty:
                    # Group by timestamp and aggregate emotions
                    grouped = user_df.groupby('timestamp').agg({
                        'Location': 'first',
                        'Emotion': lambda x: ', '.join([f"{x.tolist().count(e)} {e}" for e in set(x)]),
                        'timestamp': 'first'
                    }).reset_index(drop=True)
                    
                    # Add index starting from 1
                    grouped.index = grouped.index + 1
                    
                    # Display table on top, chart on bottom
                    st.markdown("**Records**")
                    st.dataframe(
                        grouped[["Location", "Emotion", "timestamp"]].rename(columns={"timestamp": "Time"}),
                        use_container_width=True
                    )
                    
                    st.markdown("**Emotion Distribution**")
                    fig = px.pie(user_df, names="Emotion", title="Your Emotion Distribution")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No history records found for your account.")
            else:
                st.info("No history records found.")
        else:
            st.info("No history file found.")
    except Exception as e:
        st.error(f"Error loading history: {e}")

# ----------------- Login/Signup Pages -----------------
def login_page():
    st.title("👁‍🗨 Perspēct")
    st.subheader("🕵️‍♂️ Sign In")
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
    st.title("👁‍🗨 Perspēct")
    st.subheader("🕵️‍♂️ Sign Up")
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
    
    st.title("👁‍🗨 Perspēct")
    st.caption("Upload a photo to detect facial emotions and estimate location.")
    
    # Show history if toggled, otherwise show regular tabs
    if st.session_state.get('show_history', False):
        show_user_history(username)
    else:
        tabs = st.tabs(["🏠 Home", "🗺️ Location Map"])

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

# ----------------- Run App -----------------
if __name__ == "__main__":
    # Initialize session state variables
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "show_signup" not in st.session_state:
        st.session_state.show_signup = False
    if "username" not in st.session_state:
        st.session_state.username = ""
    if "show_history" not in st.session_state:
        st.session_state.show_history = False

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
