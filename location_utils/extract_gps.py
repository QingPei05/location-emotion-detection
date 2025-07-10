# location_utils/extract_gps.py
import logging
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_gps(image_path):
    """Extract GPS information from image EXIF data"""
    try:
       
        image = Image.open(image_path)
        exif = image.getexif() or {}
        if not exif:
            logger.info("[EXIF] No EXIF data found")
            return None

        gps_info = {}
        for tag_id, value in exif.items():
            tag = TAGS.get(tag_id)
            if tag == "GPSInfo" and isinstance(value, dict):
                for key, val in value.items():
                    decoded = GPSTAGS.get(key, key)
                    gps_info[decoded] = val

        if gps_info:
            logger.info(f"[EXIF] GPS data keys: {list(gps_info.keys())}")
            return gps_info
        else:
            logger.info("[EXIF] No GPSInfo field in EXIF data")
            return None

    except Exception as e:
        logger.error(f"[EXIF ERROR] {e}")
        return None

            
        
def convert_gps(gps_info):
    
    try:
        def _safe_convert(coord, ref):
            """Safely convert coordinates handling different formats"""
            # Handle direct decimal values
            if isinstance(coord, (int, float)):
                return coord if ref in ['N', 'E'] else -coord
            
            # Handle tuple/list formats
            if isinstance(coord, (tuple, list)):
                if len(coord) == 3:  # Standard degrees, minutes, seconds
                    deg, minute, sec = coord
                    # Handle different fraction formats
                    if isinstance(deg, tuple) and len(deg) == 2:
                        deg_val = deg[0] / deg[1]
                    else:
                        deg_val = float(deg)
                    
                    if isinstance(minute, tuple) and len(minute) == 2:
                        min_val = minute[0] / minute[1]
                    else:
                        min_val = float(minute)
                    
                    if isinstance(sec, tuple) and len(sec) == 2:
                        sec_val = sec[0] / sec[1]
                    else:
                        sec_val = float(sec)
                    
                    result = deg_val + min_val/60 + sec_val/3600
                    
                elif len(coord) == 2:  # Only degrees and minutes
                    deg, minute = coord
                    if isinstance(deg, tuple) and len(deg) == 2:
                        deg_val = deg[0] / deg[1]
                    else:
                        deg_val = float(deg)
                    
                    if isinstance(minute, tuple) and len(minute) == 2:
                        min_val = minute[0] / minute[1]
                    else:
                        min_val = float(minute)
                    
                    result = deg_val + min_val/60
                    
                elif len(coord) == 1:  # Decimal degrees
                    if isinstance(coord[0], tuple) and len(coord[0]) == 2:
                        result = coord[0][0] / coord[0][1]
                    else:
                        result = float(coord[0])
                else:
                    return None
                    
                return result if ref in ['N', 'E'] else -result
            
            return None
        
        # Check for required GPS fields
        required = ["GPSLatitude", "GPSLatitudeRef", "GPSLongitude", "GPSLongitudeRef"]
        missing = [k for k in required if k not in gps_info]
        if missing:
            logger.info(f"[CONVERT] Missing GPS fields: {missing}")
            return None
        
        lat = _safe_convert(gps_info["GPSLatitude"], gps_info["GPSLatitudeRef"])
        lon = _safe_convert(gps_info["GPSLongitude"], gps_info["GPSLongitudeRef"])
        
        if lat is None or lon is None:
            logger.info("[CONVERT] Failed to convert coordinates")
            return None
            
        # Validate coordinate ranges
        if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            logger.info(f"[CONVERT] Invalid coordinates: lat={lat}, lon={lon}")
            return None
            
        logger.info(f"[CONVERT] Success: lat={lat:.6f}, lon={lon:.6f}")
        return (round(lat, 6), round(lon, 6))
        
    except Exception as e:
        logger.error(f"[CONVERT ERROR] {e}")
        return None
