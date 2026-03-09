import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from rapidocr_onnxruntime import RapidOCR
import re
import os

# --- Configuration ---
st.set_page_config(page_title="🐄 Cattle Eartag detector", layout="wide")

MISHAP_MAP = {
    "|": "1", "I": "1", "l": "1", "[": "1", "]": "1", "(": "1", ")": "1",
    "O": "0", "o": "0", "S": "5", "s": "5", "B": "8", "G": "6"
}

def load_models():
    """Load YOLO detector and RapidOCR recognizer."""
    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, 'cow_eartag_yolov8n_100ep_clean_best.pt')
    if not os.path.exists(model_path):
        st.error("Model file 'cow_eartag_yolov8n_100ep_clean_best.pt' not found.")
        st.stop()
    return YOLO(model_path), RapidOCR()

@st.cache_resource
def get_detector_recognizer():
    """Cache the models to avoid reloading on every rerun."""
    return load_models()

detector, recognizer = get_detector_recognizer()

def clean_and_format(raw_text):
    """Applies mishap mapping and keeps only digits."""
    text = raw_text.strip()
    for char, replacement in MISHAP_MAP.items():
        text = text.replace(char, replacement)
    # Remove any remaining non-digit characters
    return re.sub(r'\D', '', text)

def get_largest_text_by_pixels(ocr_result):
    """
    Filters OCR results to only get the largest text block by pixel area.
    Returns the text with the largest bounding box area.
    """
    if not ocr_result:
        return None
    
    largest_text = None
    largest_area = 0
    
    for line in ocr_result:
        box = line[0]  # Format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        text = line[1]
        
        # Calculate bounding box area
        x_coords = [p[0] for p in box]
        y_coords = [p[1] for p in box]
        width = max(x_coords) - min(x_coords)
        height = max(y_coords) - min(y_coords)
        area = width * height
        
        # Keep track of the largest text block
        if area > largest_area:
            largest_area = area
            largest_text = text
    
    return largest_text if largest_text else None

def get_largest_bbox(results):
    if not results or len(results) == 0:
        return None
    
    largest_box = None
    largest_area = 0
    
    for r in results:
        for box in r.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box)
            area = (x2 - x1) * (y2 - y1)
            
            if area > largest_area:
                largest_area = area
                largest_box = (x1, y1, x2, y2)
    
    return largest_box

def get_bottom_id(crop):
    """Finds text in the bottom half of the crop and cleans it."""
    # RapidOCR returns: [ [ [box], text, confidence ], ... ]
    result, _ = recognizer(crop)
    if not result:
        return None

    height = crop.shape[0]
    candidates = []

    for line in result:
        box = line[0]  # Format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        text = line[1]
        
        # Calculate the vertical center of this specific text block
        y_coords = [p[1] for p in box]
        text_center_y = sum(y_coords) / len(y_coords)

        # Heuristic: If text is in the bottom 60% of the tag, it's likely the Large ID
        if text_center_y > (height * 0.4):
            cleaned = clean_and_format(text)
            if cleaned:
                candidates.append(cleaned)

    # Join multiple lines if the ID is split, or return the longest numeric string
    return "".join(candidates) if candidates else None

def get_largest_id_by_pixels(crop):
    """
    Extracts the largest text block by pixel area from the crop and cleans it.
    Returns the ID of the largest number pixels in the dataset.
    """
    result, _ = recognizer(crop)
    if not result:
        return None
    
    # Get the largest text block by pixel area
    largest_text = get_largest_text_by_pixels(result)
    
    if largest_text:
        cleaned = clean_and_format(largest_text)
        return cleaned if cleaned else None
    
    return None

# --- Main UI ---
st.title("Cattle Ear tag detector")

uploaded_file = st.file_uploader("Upload Tag Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load the image
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    
    # Display uploaded image
    st.subheader("Uploaded Image")
    st.image(image)
    
    # Run detection
    results = detector(img_array, conf=0.4)
    
    # Get only the largest bounding box
    largest_box = get_largest_bbox(results)
    
    st.subheader("Extracted Large IDs")
    found_any = False
    tag_crops = []

    if largest_box:
        x1, y1, x2, y2 = largest_box
        
        # Crop the tag
        tag_crop = img_array[y1:y2, x1:x2]
        if tag_crop.size > 0:
            # Pre-processing to improve dark-on-yellow contrast
            gray = cv2.cvtColor(tag_crop, cv2.COLOR_RGB2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            processed = clahe.apply(gray)

            # Extract largest text by pixels
            large_id = get_largest_id_by_pixels(processed)
            
            if large_id:
                st.markdown(f"### `{large_id}`")
                tag_crops.append({
                    'id': large_id,
                    'crop': tag_crop,
                    'idx': 1
                })
                found_any = True
    
    if found_any:
        st.subheader("Tag Crop (Largest Detection)")
        st.image(tag_crops[0]['crop'], caption=f"Largest Tag - ID: {tag_crops[0]['id']}")
    
    if not found_any:
        st.warning("No large IDs could be clearly extracted.")
else:
    st.info("Upload an image to begin")
