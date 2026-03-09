import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from rapidocr_onnxruntime import RapidOCR
import re
import os

# --- Configuration ---
st.set_page_config(page_title="Large ID Extractor", layout="centered")

MISHAP_MAP = {
    "|": "1", "I": "1", "l": "1", "[": "1", "]": "1", "(": "1", ")": "1",
    "O": "0", "o": "0", "S": "5", "s": "5", "B": "8", "G": "6"
}

@st.cache_resource
def load_models():
    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, 'best.pt')
    if not os.path.exists(model_path):
        st.error("Model file 'best.pt' not found.")
        st.stop()
    return YOLO(model_path), RapidOCR()

detector, recognizer = load_models()

def clean_and_format(raw_text):
    """Applies mishap mapping and keeps only digits."""
    text = raw_text.strip()
    for char, replacement in MISHAP_MAP.items():
        text = text.replace(char, replacement)
    # Remove any remaining non-digit characters
    return re.sub(r'\D', '', text)

def get_bottom_id(crop):
    """Finds text in the bottom half of the crop and cleans it."""
    # RapidOCR returns: [ [ [box], text, confidence ], ... ]
    result, _ = recognizer(crop)
    if not result:
        return None

    height, _, _ = crop.shape
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

# --- Main UI ---
st.title("🆔 Large ID Extractor Only")
uploaded_file = st.file_uploader("Upload Tag Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    
    # 1. Detection
    results = detector(img_array, conf=0.4)
    
    st.subheader("Extracted Large IDs")
    found_any = False

    for r in results:
        for box in r.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box)
            
            # 2. Crop the tag
            tag_crop = img_array[y1:y2, x1:x2]
            if tag_crop.size == 0: continue
            
            # Pre-processing to improve dark-on-yellow contrast
            gray = cv2.cvtColor(tag_crop, cv2.COLOR_RGB2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            processed = clahe.apply(gray)

            # 3. Extract only the bottom text
            large_id = get_bottom_id(processed)
            
            if large_id:
                st.markdown(f"### `{large_id}`")
                found_any = True
    
    if not found_any:
        st.warning("No large IDs could be clearly extracted.")
