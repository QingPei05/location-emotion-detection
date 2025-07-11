# location_utils/landmark.py
import logging
from typing import Optional,Tuple  
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
    logger.info("Loading CLIP processor and model...")  
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    return processor, model


clip_processor, clip_model = load_models() 

# Predefined landmarks with name, city, latitude, longitude
LANDMARK_KEYWORDS = {
    #Malaysia landmarks
    "petronas towers": ["Petronas Twin Towers", "Kuala Lumpur", 3.1579, 101.7116],
    "klcc": ["Kuala Lumpur City Centre", "Kuala Lumpur", 3.1586, 101.7145],
    "kl tower": ["KL Tower", "Kuala Lumpur", 3.1528, 101.7039],
    "batu caves": ["Batu Caves", "Selangor", 3.2379, 101.6831],
    "putrajaya pink mosque": ["Putra Mosque", "Putrajaya", 2.9360, 101.6895],
    "kuala lumpur blue mosque": ["Sultan Salahuddin Abdul Aziz Mosque", "Kuala Lumpur", 3.0788, 101.5031],
    "masjid putra": ["Putra Mosque", "Putrajaya", 2.9360, 101.6895],
    "iron mosque": ["Tuanku Mizan Zainal Abidin Mosque", "Putrajaya", 2.9266, 101.6804],
    "sultan abdul samad building": ["Sultan Abdul Samad Building", "Kuala Lumpur", 3.1466, 101.6945],
    "istana negara": ["Istana Negara", "Kuala Lumpur", 3.1339, 101.6842],
    "malacca straits mosque": ["Masjid Selat Melaka", "Malacca", 2.1885, 102.2497],
    "george town street art": ["George Town Street Art", "Penang", 5.4141, 100.3288],
    "komtar": ["Komtar Tower", "Penang", 5.4143, 100.3288],
    "kek lok si": ["Kek Lok Si Temple", "Penang", 5.3991, 100.2736],
    "penang hill": ["Penang Hill", "Penang", 5.4163, 100.2766],
    "langkawi sky bridge": ["Langkawi Sky Bridge", "Langkawi", 6.3847, 99.6636],
    "gunung mat cincang": ["Gunung Mat Cincang", "Langkawi", 6.3790, 99.6652],
    "genting highlands": ["Genting Highlands", "Pahang", 3.4221, 101.7934],
    "legoland malaysia": ["Legoland Malaysia", "Johor", 1.4274, 103.6315],
    "mount kinabalu": ["Mount Kinabalu", "Sabah", 6.0755, 116.5583],
    "kinabalu park": ["Kinabalu Park", "Sabah", 6.0456, 116.6864],
    "menara alor setar": ["Alor Setar Tower", "Kedah", 6.1219, 100.3716],
    "penang bridge": ["Penang Bridge", "Penang", 5.3364, 100.3606],
    "a famosa": ["A Famosa", "Melaka", 2.1912, 102.2501],
    "sabah state mosque": ["Sabah State Mosque", "Kota Kinabalu", 5.9576, 116.0654],
    "gunung kinabalu": ["Mount Kinabalu", "Sabah", 6.0754, 116.5584],

    # Asia
    "Great Wall": ["Great Wall of China", "China", 40.4319, 116.5704],
    "burj khalifa": ["Burj Khalifa", "Dubai", 25.1972, 55.2744],
    "taipei 101": ["Taipei 101", "Taipei", 25.0330, 121.5654],
    "marina bay sands": ["Marina Bay Sands", "Singapore", 1.2834, 103.8607],

    # Europe
    "big ben": ["Big Ben", "London", 51.5007, -0.1246],
    "louvre": ["Louvre Museum", "Paris", 48.8606, 2.3376],
    "sagrada familia": ["Sagrada Família", "Barcelona", 41.4036, 2.1744],
    "leaning tower of pisa": ["Leaning Tower of Pisa", "Pisa", 43.7230, 10.3966],
    "piazza dei miracoli":    ["Piazza dei Miracoli",    "Pisa", 43.7230, 10.3966],

    # America
    "golden gate bridge": ["Golden Gate Bridge", "San Francisco", 37.8199, -122.4783],
    "times square": ["Times Square", "New York", 40.7580, -73.9855],
    "hollywood sign": ["Hollywood Sign", "Los Angeles", 34.1341, -118.3215],
    "Statue of Liberty": ["Statue of Liberty", "New York", 40.6892, -74.0445],

    # Others
    "machu picchu": ["Machu Picchu", "Peru", -13.1631, -72.5450],
    "christ the redeemer": ["Christ the Redeemer", "Rio de Janeiro", -22.9519, -43.2105],
    "opera house": ["Sydney Opera House", "Sydney", -33.8568, 151.2153],
    "sydney opera house": ["Sydney Opera House", "Sydney", -33.8568, 151.2153],
    "Eiffel Tower": ["Eiffel Tower", "Paris", 48.8584, 2.2945],
    "taj mahal": ["Taj Mahal", "Agra", 27.1751, 78.0421]
}

OVERPASS_URL = "http://overpass-api.de/api/interpreter"

def detect_landmark(
    image_path: str,
    threshold: float = 0.15,
    top_k: int = 5
) -> Optional[str]:
    """
    Use CLIP to match the image against predefined landmarks.
    Returns the matched keyword (lowercased) if score >= threshold, else None.
    """
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        keywords = list(LANDMARK_KEYWORDS.keys())

        # Text tokenization with padding/truncation
        text_inputs = clip_processor.tokenizer(
            keywords,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        # Image feature extraction
        image_inputs = clip_processor.feature_extractor(
            images=image,
            return_tensors="pt"
        )
        # Merge inputs
        inputs = {**text_inputs, **image_inputs}

        # Forward pass
        with torch.no_grad():
            outputs = clip_model(**inputs)
            logits = outputs.logits_per_image  # shape (1, len(keywords))
            probs = logits.softmax(dim=1).cpu().numpy().flatten()

        # Top-k for debug
        top_idxs = probs.argsort()[::-1][:top_k]
        for rank, idx in enumerate(top_idxs, start=1):
            logger.info(f"CLIP rank {rank}: {keywords[idx]} -> {probs[idx]:.4f}")

        best_idx = int(top_idxs[0])
        best_score = float(probs[best_idx])
        best_name = keywords[best_idx]

        if best_score >= threshold:
            logger.info(f"[CLIP MATCH] {best_name} ({best_score:.3f})")
            return best_name.lower()
        else:
            logger.info(
                f"[CLIP LOW CONFIDENCE] best={best_name} ({best_score:.3f}), threshold={threshold}"
            )
            return None
    except Exception as e:
        logger.error(f"[CLIP ERROR] {e}")
        return None


def query_landmark_coords(
    landmark_name: str
) -> Tuple[Optional[Tuple[float, float]], str]:
    """
    Given a landmark keyword, return (lat, lon) and source.
    First checks predefined dict; if missing, queries Overpass API.
    """
    key = landmark_name.lower()
    if key in LANDMARK_KEYWORDS:
        _, _, lat, lon = LANDMARK_KEYWORDS[key]
        return (lat, lon), "Predefined"

    # Build Overpass QL
    query = f"""
    [out:json][timeout:25];
    (
      node["name"~"{landmark_name}",i];
      way["name"~"{landmark_name}",i];
    );
    out center;
    """

    # Try up to 3 times
    for attempt in range(1, 4):
        try:
            resp = requests.post(OVERPASS_URL, data=query, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            elements = data.get("elements", [])
            if elements:
                elem = elements[0]
                if "center" in elem:
                    lat = elem["center"]["lat"]
                    lon = elem["center"]["lon"]
                    return (lat, lon), "Overpass"
                elif "lat" in elem and "lon" in elem:
                    return (elem["lat"], elem["lon"]), "Overpass"
            logger.warning(f"[OVERPASS] No elements found (attempt {attempt})")
        except Exception as e:
            logger.warning(f"[OVERPASS attempt {attempt}] {e}")

    return None, "No coordinates available"
