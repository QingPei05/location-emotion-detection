import pandas as pd
import os
from datetime import datetime
from configs.settings import settings

def save_history(data: dict) -> bool:
    """Save detection results to history CSV"""
    try:
        os.makedirs(settings.DATA_DIR, exist_ok=True)
        
        new_entry = pd.DataFrame([{
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "username": data.get("username", "anonymous"),
            "emotion": data["emotion"],
            "confidence": data["confidence"],
            "face_confidence": data.get("face_confidence", 0),
            "image_size": data.get("image_size", "0x0"),
            "detector": data.get("detector", "unknown"),
            "models": ",".join(data.get("models", []))
        }])
        
        if os.path.exists(settings.HISTORY_FILE):
            history = pd.read_csv(settings.HISTORY_FILE)
            history = pd.concat([history, new_entry], ignore_index=True)
            # Keep only the most recent records
            history = history.tail(settings.MAX_HISTORY_RECORDS)
        else:
            history = new_entry
        
        history.to_csv(settings.HISTORY_FILE, index=False)
        return True
    except Exception as e:
        print(f"Error saving history: {str(e)}")
        return False

def load_history(username: str = None) -> pd.DataFrame:
    """Load history records, optionally filtered by username"""
    try:
        if not os.path.exists(settings.HISTORY_FILE):
            return pd.DataFrame()
        
        history = pd.read_csv(settings.HISTORY_FILE)
        if username:
            history = history[history["username"] == username]
        return history.sort_values("timestamp", ascending=False)
    except Exception as e:
        print(f"Error loading history: {str(e)}")
        return pd.DataFrame()
