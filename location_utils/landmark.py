# location_utils/landmark.py
import logging
from typing import Optional, Tuple
import streamlit as st
import requests
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@st.cache_resource  
def load_models():
    """Load and cache CLIP models for better performance"""
    logger.info("Loading optimized CLIP models...")
    # Using smaller model variant for faster inference
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
    return processor, model

# Preload models at module level
clip_processor, clip_model = load_models()

# Predefined landmarks with name, city, latitude, longitude
LANDMARK_KEYWORDS = {
    # Malaysia landmarks
    "petronas towers": ["Petronas Twin Towers", "Kuala Lumpur", 3.1579, 101.7116],
    "klcc": ["Kuala Lumpur City Centre", "Kuala Lumpur", 3.1586, 101.7145],
    "kl tower": ["KL Tower", "Kuala Lumpur", 3.1528, 101.7039],
    "batu caves": ["Batu Caves", "Selangor", 3.2379, 101.6831],
    "putrajaya pink mosque": ["Putra Mosque", "Putrajaya", 2.9360, 101.6895],
    # ... (keep all your existing landmarks)
}

OVERPASS_URL = "http://overpass-api.de/api/interpreter"

def detect_landmark(
    image_path: str,
    threshold: float = 0.15,
    top_k: int = 3  # Reduced from original 5 for faster processing
) -> Optional[str]:
    """
    Optimized landmark detection using CLIP model.
    
    Args:
        image_path: Path to the image file
        threshold: Confidence threshold for accepting a match
        top_k: Number of top predictions to consider
        
    Returns:
        Matched landmark name (lowercase) or None if no confident match
    """
    try:
        # Load and optimize image
        image = Image.open(image_path).convert("RGB")
        image = image.resize((224, 224))  # Standard size for faster processing
        
        # Prepare inputs using preloaded processor
        inputs = clip_processor(
            text=list(LANDMARK_KEYWORDS.keys()),
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        # Model inference
        with torch.no_grad():
            outputs = clip_model(**inputs)
            logits = outputs.logits_per_image
            probs = logits.softmax(dim=1).cpu().numpy().flatten()

        # Get top predictions
        top_idxs = probs.argsort()[::-1][:top_k]
        for rank, idx in enumerate(top_idxs, start=1):
            landmark_name = list(LANDMARK_KEYWORDS.keys())[idx]
            logger.info(f"CLIP rank {rank}: {landmark_name} -> {probs[idx]:.4f}")

        best_idx = top_idxs[0]
        best_score = probs[best_idx]
        best_name = list(LANDMARK_KEYWORDS.keys())[best_idx]

        if best_score >= threshold:
            logger.info(f"[CLIP MATCH] {best_name} ({best_score:.3f})")
            return best_name.lower()
        
        logger.info(f"[CLIP LOW CONFIDENCE] best={best_name} ({best_score:.3f}), threshold={threshold}")
        return None

    except Exception as e:
        logger.error(f"[CLIP ERROR] {str(e)}")
        return None

def query_landmark_coords(
    landmark_name: str
) -> Tuple[Optional[Tuple[float, float]], str]:
    """
    Get coordinates for a landmark with fallback to Overpass API.
    
    Args:
        landmark_name: Name of the landmark to search for
        
    Returns:
        Tuple of (coordinates, source) or (None, error message)
    """
    # First check predefined landmarks
    key = landmark_name.lower()
    if key in LANDMARK_KEYWORDS:
        _, _, lat, lon = LANDMARK_KEYWORDS[key]
        return (lat, lon), "Predefined"

    # Fallback to Overpass API
    query = f"""
    [out:json][timeout:15];
    (
      node["name"~"{landmark_name}",i];
      way["name"~"{landmark_name}",i];
    );
    out center;
    """

    for attempt in range(1, 4):
        try:
            resp = requests.post(OVERPASS_URL, data=query, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            
            if elements := data.get("elements", []):
                elem = elements[0]
                if "center" in elem:
                    return (elem["center"]["lat"], elem["center"]["lon"]), "Overpass"
                elif "lat" in elem and "lon" in elem:
                    return (elem["lat"], elem["lon"]), "Overpass"
                
            logger.warning(f"[OVERPASS] No valid elements found (attempt {attempt})")
        except Exception as e:
            logger.warning(f"[OVERPASS attempt {attempt}] Error: {str(e)}")
            if attempt < 3:
                import time
                time.sleep(1)  # Brief delay before retry

    return None, "No coordinates available"
