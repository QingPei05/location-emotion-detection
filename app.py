import os, streamlit as st
import sys, subprocess
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from datetime import datetime
import random
import plotly.express as px
from emotion_utils.detector import EmotionDetector
import hashlib
import tempfile
from location_utils.extract_gps import extract_gps, convert_gps
from location_utils.geocoder import get_address_from_coords
from location_utils.landmark import load_models, detect_landmark, query_landmark_coords, LANDMARK_KEYWORDS

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

# Load CLIP models once
processor, clip_model = load_models()

def save_history(username, emotions, confidences, location):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    records = []
    for i, (emo, conf) in enumerate(zip(emotions, confidences)):
        records.append([username, location, emo, conf, now])
    
    df = pd.DataFrame(records, columns=["username", "Location", "Emotion", "Confidence", "timestamp"])
    try:
        if os.path.exists("history.csv"):
            prev = pd.read_csv("history.csv")
            df = pd.concat([prev, df], ignore_index=True)
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
        if st.button("⬅ Back to Main", key="back_button"):
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
                    
                    # Rename timestamp to Time for display
                    grouped_display = grouped.rename(columns={"timestamp": "Time"})
                    
                    # Display table with checkboxes in last column
                    st.markdown("**📝 Records**")
                    
                    # Initialize selection state if not exists
                    if 'select_all_state' not in st.session_state:
                        st.session_state.select_all_state = False
                    
                    # Add select column with current selection state
                    grouped_display['Select'] = st.session_state.select_all_state
                    
                    # Display non-editable table with checkboxes
                    edited_df = st.data_editor(
                        grouped_display[["Location", "Emotion", "Time", "Select"]],
                        disabled=["Location", "Emotion", "Time"],
                        hide_index=True,
                        use_container_width=True
                    )
                    
                    # Add select all and delete buttons on the right
                    col1, col2 = st.columns([4, 1])
                    with col2:
                        select_all = st.checkbox("Select All", key="select_all", value=st.session_state.select_all_state)
                        if select_all != st.session_state.select_all_state:
                            st.session_state.select_all_state = select_all
                            st.rerun()
                        
                        if st.button("🗑️ Delete", key="delete_button"):
                            # Get indices of selected rows
                            selected_indices = edited_df.index[edited_df['Select']].tolist()
                            if selected_indices:
                                # Safely get the timestamps to delete
                                try:
                                    timestamps_to_delete = grouped.loc[selected_indices, "timestamp"].tolist()
                                    # Filter out the deleted records
                                    df = df[~((df["username"] == username) & (df["timestamp"].isin(timestamps_to_delete)))]
                                    # Save back to CSV
                                    df.to_csv("history.csv", index=False)
                                    st.success("Selected records deleted successfully!")
                                    st.session_state.select_all_state = False
                                    st.rerun()
                                except KeyError:
                                    st.error("Error: Could not find selected records to delete")
        
                    # Add spacing between table and chart
                    st.markdown("<br><br>", unsafe_allow_html=True)
                    st.markdown("**📊 Emotion Distribution**")
                    
                    # Create columns for the selection and chart
                    col_select, col_chart = st.columns([2, 5])
                    
                    with col_select:
                        # Add record selection for chart
                        records = grouped["timestamp"].tolist()
                        records.insert(0, "All")  # Add "All" option
                        selected_record = st.selectbox("Select record to view:", 
                                                     ["All"] + [str(ts) for ts in grouped["timestamp"].tolist()], 
                                                     index=0)

                        # Filter data based on selection
                        if selected_record == "All":
                            chart_data = user_df
                        else:
                            chart_data = user_df[user_df["timestamp"] == selected_record]

                    with col_chart:
                        # Display chart with simplified title
                        fig = px.pie(chart_data, names="Emotion")
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
    
    with st.form("login_form"):
        username = st.text_input("Username", label_visibility="collapsed", placeholder="Username")
        password = st.text_input("Password", type="password", label_visibility="collapsed", placeholder="Password")
        
        # Buttons side by side with Sign Up pushed to right
        cols = st.columns([3, 1])  # 3:1 ratio for left vs right space
        with cols[0]:
            login_submitted = st.form_submit_button("Log In")
        with cols[1]:
            signup_clicked = st.form_submit_button("Sign Up →")
        
        if login_submitted:
            if authenticate(username, password):
                st.session_state["logged_in"] = True
                st.session_state["username"] = username
                st.rerun()
            else:
                st.error("Invalid username or password")
        elif signup_clicked:
            st.session_state["show_signup"] = True
            st.rerun()

def signup_page():
    st.title("👁‍🗨 Perspēct")
    st.subheader("🕵️‍♂️ Sign Up")
    
    with st.form("signup_form"):
        username = st.text_input("Choose a username", label_visibility="collapsed", placeholder="Username")
        password = st.text_input("Choose a password", type="password", label_visibility="collapsed", placeholder="Password")
        confirm_password = st.text_input("Confirm password", type="password", label_visibility="collapsed", placeholder="Confirm Password")
        
        # Buttons side by side with Back pushed to right
        cols = st.columns([3, 1])  # 3:1 ratio for left vs right space
        with cols[0]:
            register_submitted = st.form_submit_button("Register")
        with cols[1]:
            back_clicked = st.form_submit_button("← Back")
        
        if register_submitted:
            if not username or not password or not confirm_password:
                st.error("Username and password are required!")
            elif password != confirm_password:
                st.error("Passwords don't match")
            elif register_user(username, password):
                st.success("Registration successful! Please sign in.")
                st.session_state["show_signup"] = False
                st.rerun()
            else:
                st.error("Username already exists or registration failed")
        elif back_clicked:
            st.session_state["show_signup"] = False
            st.rerun()

# ----------------- Main App -----------------
def main_app():
    username = st.session_state.get("username", "")
    sidebar_design(username)

    if "coords_result" not in st.session_state:
        st.session_state.coords_result = None
    if "location_method" not in st.session_state:
        st.session_state.location_method = ""
    
    st.markdown("""
    <div style="
        background-color: #f8f8e7;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    ">
        <h1 style="color: #333333; font-family: 'Segoe UI', sans-serif;">👁‍🗨 Perspēct</h1>
        <p style="font-size: 1.1rem; color: #555555; font-family: 'Segoe UI', sans-serif;">
            Upload a photo to detect facial emotions and estimate location.
        </p>
    </div>
    """, unsafe_allow_html=True)

    
    # Show history if toggled, otherwise show regular tabs
    if st.session_state.get('show_history', False):
        show_user_history(username)
    else:
        tabs = st.tabs(["🏠 Home", "🗺️ Location Map"])

        with tabs[0]:
            uploaded_file = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "png"])
            if uploaded_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    temp_path = tmp_file.name
                    
                try:
                    image = Image.open(uploaded_file).convert("RGB")
                    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    detector = EmotionDetector()
                    detections = detector.detect_emotions(img)
                    detected_img = detector.draw_detections(img, detections)

                    location = "Unknown"
                    coords = None
                    face_word = "face" if len(detections) == 1 else "faces"

                    # 1) Try EXIF GPS
                    gps_info = extract_gps(temp_path)
                    if gps_info:
                        coords = convert_gps(gps_info)
                        if coords:
                            st.session_state.coords_result = coords
                            st.session_state.location_method = "GPS Metadata"
                            location = get_address_from_coords(coords)
                            
                    # 2) Fallback to CLIP landmark
                    if coords is None:
                        landmark = detect_landmark(temp_path, threshold=0.15, top_k=5)
                        if landmark:
                            st.write(f"🔍 CLIP predicted landmark: **{landmark}**")
                            coords_loc, source = query_landmark_coords(landmark)
                            if coords_loc:
                                st.session_state.coords_result = coords_loc
                                st.session_state.location_method = f"Landmark ({source})"
                                addr = get_address_from_coords(coords_loc)
                                if addr not in (
                                    "Unknown location",
                                    "Geocoding service unavailable"
                                ):  # Valid address
                                    location = addr
                                else:
                                    info = LANDMARK_KEYWORDS.get(landmark)
                                    if info:
                                        location = f"{info[0]}, {info[1]}"
                                    else:
                                        lat, lon = coords_loc
                                        location = f"{landmark.title()} ({lat:.4f}, {lon:.4f})"
                        else:
                            st.write("🔍 No landmark detected with sufficient confidence")

                except Exception as e:
                    st.error(f"❌ Something went wrong during processing: {e}")

                # Display detection results
                if detections:
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.subheader("🔍 Detection Results")
                        if detections:
                            emotions = [d["emotion"] for d in detections]
                            confidences = [d["confidence"] for d in detections]
                            
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
                            method = st.session_state.get("location_method", "")
                            st.write(
                                f"📍 Estimated Location: **{location}** "
                            )
                            save_history(username, emotions, confidences, location)
                        else:
                            st.warning("No faces were detected in the uploaded image.")
                    with col2:
                        t1, t2 = st.tabs(["Original Image", "Processed Image"])
                        with t1:
                            st.image(image, use_container_width=True)
                        with t2:
                            st.image(detected_img, channels="BGR", use_container_width=True,
                                    caption=f"Detected {len(detections)} {face_word}")
                else:
                    st.warning("No faces were detected in the uploaded image.")

                    # Cleanup temp file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

        with tabs[1]:
            st.subheader("🗺️ Detected Location Map")
            coords_result = st.session_state.get("coords_result", None)
            method = st.session_state.get("location_method", "")
            if coords_result:
                lat, lon = coords_result
                map_df = pd.DataFrame({"lat": [lat], "lon": [lon]})
                st.map(map_df)
                st.caption(f"Source: {method}")
            else:
                st.info("No detected location to display on map.")
            
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
