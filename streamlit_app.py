import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from rapidocr_onnxruntime import RapidOCR
import re
import os
import zipfile
import io

# --- 1. Configuration & Constants ---
st.set_page_config(page_title="Cattle Ear Tag ID Extractor", layout="wide")

MISHAP_MAP = {
    "|": "1", "I": "1", "l": "1", "[": "1", "]": "1", "(": "1", ")": "1",
    "O": "0", "o": "0", "S": "5", "s": "5", "B": "8", "G": "6"
}

# --- 2. Model Loading ---
@st.cache_resource
def load_models():
    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, 'best.pt')
    
    if not os.path.exists(model_path):
        st.error(f"Error: 'best.pt' not found at {model_path}. Please ensure weights are in the root directory.")
        st.stop()
        
    # Initialize YOLO and RapidOCR
    return YOLO(model_path), RapidOCR()

detector, recognizer = load_models()

# --- 3. Helper Functions ---
def clean_and_format(raw_text):
    """Fixes common character mishaps and strips non-digits."""
    text = raw_text.strip()
    for char, replacement in MISHAP_MAP.items():
        text = text.replace(char, replacement)
    # Keep only digits for the final ID
    return re.sub(r'\D', '', text)

def get_bottom_id(tag_crop):
    """
    Analyzes text position within the tag.
    Filters for text in the bottom 60% of the crop to isolate Large IDs.
    """
    result, _ = recognizer(tag_crop)
    if not result:
        return None

    height = tag_crop.shape[0]
    candidates = []

    for line in result:
        box, text, conf = line[0], line[1], line[2]
        
        # Calculate vertical center of the text block
        y_coords = [p[1] for p in box]
        text_center_y = sum(y_coords) / len(y_coords)

        # Geometric filter: Only process text in the lower half of the tag
        if text_center_y > (height * 0.4):
            cleaned = clean_and_format(text)
            if cleaned:
                candidates.append(cleaned)

    return "".join(candidates) if candidates else None

def process_and_display(img_array, file_name):
    """Runs detection and OCR, then displays results in Streamlit."""
    results = detector(img_array, conf=0.4)
    
    found_ids = []
    for r in results:
        for box in r.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box)
            
            # Crop and Pre-process
            tag_crop = img_array[y1:y2, x1:x2]
            if tag_crop.size == 0: continue
            
            # Contrast enhancement for better OCR
            gray = cv2.cvtColor(tag_crop, cv2.COLOR_RGB2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)

            # Extract Large ID
            large_id = get_bottom_id(enhanced)
            if large_id:
                found_ids.append(large_id)

    # Display results
    with st.expander(f"Results for {file_name}"):
        col1, col2 = st.columns([1, 1])
        col1.image(img_array, use_container_width=True)
        if found_ids:
            for i, id_val in enumerate(found_ids):
                col2.success(f"Tag {i+1} Large ID: **{id_val}**")
        else:
            col2.warning("No large ID detected.")

# --- 4. Main UI Logic ---
st.title("🐄 Cattle Ear Tag ID Extractor")
st.markdown("Extracts only the **large bottom numbers** from your livestock ear tags.")

uploaded_file = st.file_uploader("Upload Image or ZIP folder", type=["jpg", "jpeg", "png", "zip"])

if uploaded_file:
    # Handle ZIP uploads
    if uploaded_file.name.lower().endswith('.zip'):
        try:
            with zipfile.ZipFile(uploaded_file) as z:
                # Filter for valid image formats and ignore hidden MacOS files
                img_list = [f for f in z.namelist() if f.lower().endswith(('.png', '.jpg', '.jpeg')) 
                            and not f.startswith('__')]
                
                st.info(f"Processing {len(img_list)} images from ZIP...")
                for img_path in img_list:
                    with z.open(img_path) as f:
                        img_bytes = f.read()
                        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                        process_and_display(np.array(image), img_path)
        except zipfile.BadZipFile:
            st.error("The uploaded file is not a valid ZIP archive.")

    # Handle Single Image uploads
    else:
        image = Image.open(uploaded_file).convert("RGB")
        process_and_display(np.array(image), uploaded_file.name)
