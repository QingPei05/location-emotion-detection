import os
import sys
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from datetime import datetime
import plotly.express as px
import hashlib
import tempfile
import concurrent.futures

# Fix module import paths
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

try:
    from emotion_utils.detector import EmotionDetector
    from location_utils.extract_gps import extract_gps, convert_gps
    from location_utils.geocoder import get_address_from_coords
    from location_utils.landmark import load_models, detect_landmark, query_landmark_coords, LANDMARK_KEYWORDS
except ImportError as e:
    st.error(f"Failed to import required modules: {str(e)}")
    st.error("Please ensure your project structure is:")
    st.error("""
    your_project/
    ├── emotion_utils/
    │   ├── __init__.py
    │   ├── detector.py
    │   └── config.py
    ├── location_utils/
    │   ├── __init__.py
    │   ├── extract_gps.py
    │   ├── geocoder.py
    │   └── landmark.py
    └── app.py
    """)
    st.stop()

# ----------------- App Configuration -----------------
st.set_page_config(
    page_title="Perspēct",
    page_icon="👁‍🗨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- Cached Resources -----------------
@st.cache_resource
def load_all_models():
    """Load and cache all ML models for better performance"""
    return {
        'emotion': EmotionDetector(),
        'clip': load_models()  # CLIP models
    }

@st.cache_data(ttl=3600, show_spinner="Processing image...")
def process_image_file(uploaded_file):
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
        if os.path.exists("users.csv"):
            users = pd.read_csv("users.csv")
            if username in users["username"].values:
                return False
        
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        new_user = pd.DataFrame([[username, hashed_password]], 
                              columns=["username", "password"])
        
        if os.path.exists("users.csv"):
            new_user.to_csv("users.csv", mode='a', header=False, index=False)
        else:
            new_user.to_csv("users.csv", index=False)
            
        return True
    except Exception as e:
        print(f"Registration error: {e}")
        return False

# ----------------- Processing Functions -----------------
def process_emotion_parallel(image_data, detector):
    """Process emotions using cached image"""
    try:
        detections = detector.detect_emotions(image_data['img_bgr'])
        detected_img = detector.draw_detections(image_data['img_bgr'], detections)
        return detections, detected_img
    except Exception as e:
        st.error(f"Emotion detection failed: {str(e)}")
        return [], None

def process_location_parallel(temp_path):
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

def save_history(username, emotions, confidences, location):
    """Save detection history to CSV"""
    try:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        records = []
        for i, (emo, conf) in enumerate(zip(emotions, confidences)):
            records.append([username, location, emo, conf, now])
        
        df = pd.DataFrame(records, columns=["username", "Location", "Emotion", "Confidence", "timestamp"])
        if os.path.exists("history.csv"):
            prev = pd.read_csv("history.csv")
            df = pd.concat([prev, df], ignore_index=True)
        df.to_csv("history.csv", index=False)
    except Exception as e:
        st.error(f"Failed to save history: {e}")

# ----------------- UI Components -----------------
def gradient_card(subtitle=None):
    """Styled header card"""
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

def show_emo_detection_guide():
    """Emotion detection explanation"""
    with st.expander("ℹ️ How Emotion Detection Works", expanded=False):
        st.markdown("""
        *Detection Logic Explained:*
        - 😊 **Happy**: Smile present, cheeks raised
        - 😠 **Angry**: Eyebrows lowered, eyes wide open
        - 😐 **Neutral**: No strong facial movements
        - 😢 **Sad**: Eyebrows raised, lip corners down
        - 😲 **Surprise**: Eyebrows raised, mouth open
        - 😨 **Fear**: Eyes tense, lips stretched
        - 🤢 **Disgust**: Nose wrinkled, upper lip raised
        """)

def show_loc_detection_guide():
    """Location detection explanation"""
    with st.expander("ℹ️ How Location Detection Works", expanded=False):
        st.markdown("""
        *How It Works:*
        - First tries to extract GPS metadata from image
        - If no GPS, uses AI to recognize landmarks
        - Finally attempts to geocode any found coordinates
        """)

def sidebar_design(username):
    """Design the sidebar"""
    if username:
        st.sidebar.success(f"👤 Logged in as: {username}")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("## Quick Navigation")
    st.sidebar.markdown("- Upload and detect emotions")
    st.sidebar.markdown("- View location map")
    st.sidebar.divider()
    st.sidebar.info("Enhance your experience by ensuring clear, well-lit facial images.")
    st.sidebar.divider()
    
    if username:
        if st.sidebar.button("📜 History", key="history_button"):
            st.session_state.show_history = not st.session_state.get('show_history', False)
    
    if st.sidebar.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.show_history = False
        st.rerun()

def show_user_history(username):
    """Display user history"""
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("📜 Your History")
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("⬅ Back to Main", key="back_button"):
            st.session_state.show_history = False
            st.rerun()
    
    try:
        if os.path.exists("history.csv"):
            df = pd.read_csv("history.csv")
            if not df.empty:
                if 'username' not in df.columns:
                    df['username'] = ""
                
                user_df = df[df["username"] == username]
                
                if not user_df.empty:
                    grouped = user_df.groupby('timestamp').agg({
                        'Location': 'first',
                        'Emotion': lambda x: ', '.join([f"{x.tolist().count(e)} {e}" for e in set(x)]),
                        'timestamp': 'first'
                    }).reset_index(drop=True)
                    
                    grouped.index = grouped.index + 1
                    grouped_display = grouped.rename(columns={"timestamp": "Time"})
                    
                    st.markdown("**📝 Records**")
                    grouped_display['Select'] = st.session_state.get('select_all_state', False)
                    
                    edited_df = st.data_editor(
                        grouped_display[["Location", "Emotion", "Time", "Select"]],
                        disabled=["Location", "Emotion", "Time"],
                        hide_index=True,
                        use_container_width=True
                    )
                    
                    col1, col2 = st.columns([4, 1])
                    with col2:
                        select_all = st.checkbox("Select All", key="select_all", 
                                               value=st.session_state.get('select_all_state', False))
                        if select_all != st.session_state.get('select_all_state', False):
                            st.session_state.select_all_state = select_all
                            st.rerun()
                        
                        if st.button("🗑️ Delete", key="delete_button"):
                            selected_indices = edited_df.index[edited_df['Select']].tolist()
                            if selected_indices:
                                try:
                                    timestamps_to_delete = grouped.loc[selected_indices, "timestamp"].tolist()
                                    df = df[~((df["username"] == username) & (df["timestamp"].isin(timestamps_to_delete))]
                                    df.to_csv("history.csv", index=False)
                                    st.success("Selected records deleted successfully!")
                                    st.session_state.select_all_state = False
                                    st.rerun()
                                except KeyError:
                                    st.error("Error: Could not find selected records to delete")
        
                    st.markdown("<br><br>", unsafe_allow_html=True)
                    st.markdown("**📊 Emotion Distribution**")
                    
                    col_select, col_chart = st.columns([2, 5])
                    
                    with col_select:
                        records = grouped["timestamp"].tolist()
                        records.insert(0, "All")
                        selected_record = st.selectbox("Select record to view:", 
                                                       ["All"] + [str(ts) for ts in grouped["timestamp"].tolist()], 
                                                       index=0)

                        if selected_record == "All":
                            chart_data = user_df
                        else:
                            chart_data = user_df[user_df["timestamp"] == selected_record]

                    with col_chart:
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

def get_location_string():
    """Generate location description from session state"""
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

# ----------------- Authentication Pages -----------------
def login_page():
    """Login page UI"""
    gradient_card()
    st.subheader("🕵️‍♂️ Sign In")
    
    with st.form("login_form"):
        username = st.text_input("Username", label_visibility="collapsed", placeholder="Username")
        password = st.text_input("Password", type="password", label_visibility="collapsed", placeholder="Password")
        
        cols = st.columns([3, 1])
        with cols[0]:
            login_submitted = st.form_submit_button("Log In")
        with cols[1]:
            signup_clicked = st.form_submit_button("Sign Up →")
        
        if login_submitted:
            if authenticate(username, password):
                st.session_state.update({
                    "logged_in": True,
                    "username": username,
                    "show_signup": False
                })
                st.rerun()
            else:
                st.error("Invalid username or password")
        elif signup_clicked:
            st.session_state["show_signup"] = True
            st.rerun()

def signup_page():
    """Signup page UI"""
    gradient_card()
    st.subheader("🕵️‍♂️ Sign Up")
    
    with st.form("signup_form"):
        username = st.text_input("Choose a username", label_visibility="collapsed", placeholder="Username")
        password = st.text_input("Choose a password", type="password", label_visibility="collapsed", placeholder="Password")
        confirm_password = st.text_input("Confirm password", type="password", label_visibility="collapsed", placeholder="Confirm Password")
        
        cols = st.columns([3, 1])
        with cols[0]:
            register_submitted = st.form_submit_button("Register")
        with cols[1]:
            back_clicked = st.form_submit_button("← Back")
        
        if register_submitted:
            if not username or not password or not confirm_password:
                st.error("All fields are required!")
            elif password != confirm_password:
                st.error("Passwords don't match")
            elif register_user(username, password):
                st.success("Registration successful! Please sign in.")
                st.session_state["show_signup"] = False
                st.rerun()
            else:
                st.error("Username already exists")
        elif back_clicked:
            st.session_state["show_signup"] = False
            st.rerun()

# ----------------- Main App -----------------
def main_app():
    """Main application logic"""
    models = load_all_models()
    username = st.session_state.get("username", "")
    
    # Initialize session state
    for key in ['coords_result', 'location_method', 'landmark', 'show_history']:
        st.session_state.setdefault(key, None)

    gradient_card("Upload a photo to detect facial emotions and estimate location")
    
    if st.session_state.get('show_history', False):
        show_user_history(username)
    else:
        tabs = st.tabs(["🏠 Home", "🗺️ Location Map"])
        
        with tabs[0]:
            if uploaded_file := st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "png"]):
                image_data = process_image_file(uploaded_file)
                
                if image_data:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        temp_path = tmp_file.name
                    
                    try:
                        with st.spinner('Analyzing image...'):
                            with concurrent.futures.ThreadPoolExecutor() as executor:
                                emotion_future = executor.submit(
                                    process_emotion_parallel, 
                                    image_data, 
                                    models['emotion']
                                )
                                location_future = executor.submit(
                                    process_location_parallel,
                                    temp_path
                                )
                                
                                detections, detected_img = emotion_future.result()
                                coords, method = location_future.result()
                                
                            st.session_state.update({
                                'coords_result': coords,
                                'location_method': method,
                                'landmark': detect_landmark(temp_path) if coords else None
                            })
                            
                            if detections:
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
                                    save_history(username, emotions, confidences, location)
                                
                                with col2:
                                    tab1, tab2 = st.tabs(["Original", "Processed"])
                                    with tab1:
                                        st.image(image_data['pil_image'], use_column_width=True)
                                    with tab2:
                                        st.image(detected_img, channels="BGR", use_column_width=True,
                                               caption=f"Detected {len(detections)} {face_word}")
                            else:
                                st.warning("No faces detected in the image")
                    
                    finally:
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
        
        with tabs[1]:
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
    st.session_state.setdefault('logged_in', False)
    st.session_state.setdefault('show_signup', False)
    st.session_state.setdefault('username', "")
    st.session_state.setdefault('show_history', False)
    
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
