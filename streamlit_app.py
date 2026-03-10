import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from rapidocr_onnxruntime import RapidOCR
import re
import os

# --- Configuration ---
st.set_page_config(page_title="Cattle Eartag detector", layout="wide")

# FIXED: Corrected syntax error (added colons and values)
MISHAP_MAP = {
    "|": "1", "I": "1", "l": "1", "[": "1", "]": "1", 
    "(": "1", ")": "1", "0": "0", "O": "0", "o": "0", 
    "S": "5", "s": "5", "B": "8", "G": "6"
}

@st.cache_resource
def get_models():
    """Load and cache models."""
    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, 'cow_eartag_yolov8n_100ep_clean_best.pt')
    if not os.path.exists(model_path):
        st.error(f"Model file '{model_path}' not found.")
        st.stop()
    return YOLO(model_path), RapidOCR()

detector, recognizer = get_models()

def clean_and_format(raw_text):
    """Applies mishap mapping and keeps only digits."""
    text = raw_text.strip()
    for char, replacement in MISHAP_MAP.items():
        text = text.replace(char, replacement)
    return re.sub(r'\D', '', text)

def process_tag_ocr(crop):
    """
    Finds text in the bottom half, adds padding for better edge recognition,
    and selects the largest block by pixel area.
    """
    # 1. Add Padding (OpenCV) 
    # This ensures characters near the edge (like that '1') aren't cut off
    padding = 10
    crop = cv2.copyMakeBorder(crop, padding, padding, padding, padding, 
                              cv2.BORDER_CONSTANT, value=[255, 255, 255])

    # 2. Pre-processing for better contrast
    bgr_crop = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
    
    # Use Adaptive Thresholding to make the text solid black on white
    # This helps when the tag is dirty or in shadow
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    
    # 3. Run OCR
    result, _ = recognizer(thresh)
    if not result:
        return None

    crop_h, _ = thresh.shape[:2]
    largest_text = ""
    max_area = 0

    for line in result:
        box, text, conf = line
        
        # Calculate vertical center and area
        y_coords = [p[1] for p in box]
        x_coords = [p[0] for p in box]
        
        y_center = sum(y_coords) / 4
        width = max(x_coords) - min(x_coords)
        height = max(y_coords) - min(y_coords)
        area = width * height

        # Filter: Only bottom half + Largest block
        if y_center > (crop_h * 0.4):
            if area > max_area:
                max_area = area
                largest_text = text

    return clean_and_format(largest_text) if largest_text else None

# --- Main UI ---
st.title("Cattle Ear Tag Detector & OCR")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    viz_img = img_array.copy()
    
    results = detector(img_array, conf=0.4)
    detections = results[0].boxes.xyxy.cpu().numpy()
    
    found_tags = []
    
    for i, box in enumerate(detections):
        x1, y1, x2, y2 = map(int, box)
        crop = img_array[y1:y2, x1:x2]
        
        if crop.size == 0:
            continue
            
        tag_id = process_tag_ocr(crop)
        display_id = tag_id if tag_id else "???"
        
        # Draw on main image
        cv2.rectangle(viz_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(viz_img, f"ID: {display_id}", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        found_tags.append({"id": display_id, "crop": crop})
    
    st.subheader("Detection Result")
    st.image(viz_img)
    
    if found_tags:
        st.subheader("Individual Tag Details")
        cols = st.columns(len(found_tags))
        for idx, tag in enumerate(found_tags):
            with cols[idx]:
                st.image(tag['crop'])
                st.write(f"**Detected ID:** `{tag['id']}`")
    else:
        st.warning("No tags detected.")
