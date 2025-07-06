import os
import pandas as pd
from datetime import datetime
import cv2

def draw_detections(img, emotions, faces, confidences, config):
    """Draw detection boxes with labels"""
    output_img = img.copy()
    color_map = config["color_map"]
    
    for (x, y, w, h), emotion, confidence in zip(faces, emotions, confidences):
        color = color_map.get(emotion, (255, 255, 255))
        
        # Draw rectangle
        cv2.rectangle(output_img, (x, y), (x+w, y+h), color, 3)
        
        # Draw label
        label = f"{emotion} {confidence}%"
        cv2.putText(
            output_img,
            label,
            (x+5, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )
    
    return output_img

def save_history(username, emotion, confidence):
    """Save detection results to history"""
    os.makedirs("history", exist_ok=True)
    history_file = "history/emotion_history.csv"
    
    new_entry = pd.DataFrame([{
        "Username": username,
        "Emotion": emotion,
        "Confidence": confidence,
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }])
    
    try:
        if os.path.exists(history_file):
            history = pd.read_csv(history_file)
            history = pd.concat([history, new_entry])
        else:
            history = new_entry
        
        history.to_csv(history_file, index=False)
    except Exception as e:
        print(f"Error saving history: {e}")
