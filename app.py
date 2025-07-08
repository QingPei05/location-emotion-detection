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

        # 显示历史记录
        st.sidebar.markdown("---")
        st.sidebar.subheader("📜 Upload History")
        if os.path.exists("history.csv"):
            df = pd.read_csv("history.csv")
            user_history = df[df["username"] == username]  # 只显示该用户的历史记录
            
            if user_history.empty:
                st.sidebar.info("No upload records found.")
            else:
                edited_df = st.sidebar.data_editor(
                    user_history[["Location", "Emotion", "Confidence", "timestamp"]],
                    key="history_editor",
                    disabled=True  # 禁用编辑
                )
                # 在侧边栏显示图表
                fig = px.pie(user_history, names="Emotion", title="Emotion Distribution")
                st.sidebar.plotly_chart(fig)

        else:
            st.sidebar.info("No history file found.")
    st.sidebar.markdown("---")
    st.sidebar.markdown("## Quick Navigation")
    st.sidebar.markdown("- Upload and detect emotions")
    st.sidebar.markdown("- Visualize your emotion distribution")
    st.sidebar.divider()

# 在Main App中显示选中的记录的详细信息
if "history_editor" in st.session_state:
    selected_rows = st.session_state.history_editor["edited_rows"]
    for idx in selected_rows:
        with st.expander(f"Details for record {idx + 1}"):
            row_data = user_history.iloc[idx]
            st.image(row_data['image_path'], use_column_width=True)  # 假设您存储了图像路径
            st.write(f"Location: {row_data['Location']}")
            st.write(f"Emotion: {row_data['Emotion']}")
            st.write(f"Timestamp: {row_data['timestamp']}")
            # 显示情绪图表
            emotion_chart_data = user_history[user_history["timestamp"] == row_data['timestamp']]
            fig = px.pie(emotion_chart_data, names="Emotion", title="Emotion Distribution for this Record")
            st.plotly_chart(fig)
    
    # Add logout button
    if st.sidebar.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.rerun()

# ----------------- Login/Signup Pages -----------------
def login_page():
    st.markdown("<h1 style='text-align: center;'>AI Emotion & Location Detector</h1>", unsafe_allow_html=True)
    st.subheader("👤 Sign In")
    username = st.text_input("Username", key="login_username")
    password = st.text_input("Password", type="password", key="login_password")
    
    if st.button("Sign In"):
        if authenticate(username, password):
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
            st.rerun()
        else:
            st.error("Invalid username or password")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Sign Up"):
            st.session_state["show_signup"] = True
            st.rerun()
    with col2:
        st.button("Enter", on_click=lambda: authenticate(username, password))

def signup_page():
    st.markdown("<h1 style='text-align: center;'>AI Emotion & Location Detector</h1>", unsafe_allow_html=True)
    st.subheader("👤 Sign Up")
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
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Back to Sign In"):
            st.session_state["show_signup"] = False
            st.rerun()
    with col2:
        st.button("Enter", on_click=lambda: register_user(username, password))

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

                        # 显示每种情绪的总数
                        emotion_count = {emo: emotions.count(emo) for emo in set(emotions)}
                        total_summary = ", ".join([f"{count} {emo}" for emo, count in emotion_count.items()])
                        st.success(f"🎭 {len(detections)} faces detected. Total: {total_summary}")
                    
                        for i, (emo, conf) in enumerate(zip(emotions, confidences)):
                            st.write(f"- Face {i + 1}: {emo} ({conf}%)")
                        save_history(username, emotions, confidences, "Unknown")
                    else:
                        st.warning("No faces were detected in the uploaded image.")
                with col2:
                    st.image(detected_img, channels="BGR", use_column_width=True)

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
