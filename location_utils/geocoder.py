# location_utils/geocoder.py

import logging
import time
from typing import Tuple

from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize geocoder with rate limiting
geolocator = Nominatim(
    user_agent="geoai_app_v2",
    timeout=10
)
reverse_geocode = RateLimiter(
    geolocator.reverse,
    min_delay_seconds=1
)

def get_address_from_coords(
    coords: Tuple[float, float],
    language: str = "en"
) -> str:
    """
    Reverse-geocode a (lat, lon) tuple into a human-readable address.
    Retries up to 3 times on transient errors.
    """
    for attempt in range(3):
        try:
            location = reverse_geocode(coords, language=language)
            if location and location.address:
                logger.info(f"[GEOCODER] Success: {location.address}")
                return location.address
            else:
                logger.info("[GEOCODER] No location found")
                return "Unknown location"
        except Exception as e:
            logger.warning(f"[GEOCODER] Attempt {attempt+1} failed: {e}")
            if attempt < 2:
                time.sleep(2)

    logger.error(f"[GEOCODER] All geocoding attempts failed for {coords}")
    return "Geocoding service unavailable"
