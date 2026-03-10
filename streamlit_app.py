import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from rapidocr_onnxruntime import RapidOCR
import re
import os

# --- Configuration ---
st.set_page_config(page_title="Cattle Ear Tag Pro", layout="wide")

MISHAP_MAP = {
    "|": "1", "I": "1", "l": "1", "[": "1", "]": "1", "(": "1", ")": "1",
    "O": "0", "o": "0", "S": "5", "s": "5", "B": "8", "G": "6"
}

@st.cache_resource
def get_models():
    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, 'cow_eartag_yolov8n_100ep_clean_best.pt')
    return YOLO(model_path), RapidOCR()

detector, recognizer = get_models()

def clean_and_format(raw_text):
    text = raw_text.strip()
    for char, replacement in MISHAP_MAP.items():
        text = text.replace(char, replacement)
    return re.sub(r'\D', '', text)

def process_tag_ocr(crop):
    """
    Standardizes the crop for OCR using Adaptive Thresholding.
    """
    bgr_crop = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
    
    # Improves contrast for dirty or shadowed tags (common in the dataset)
    enhanced = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
    
    result, _ = recognizer(enhanced)
    if not result:
        return None

    crop_h, _ = enhanced.shape[:2]
    largest_text = ""
    max_area = 0

    for line in result:
        box, text, conf = line
        y_coords = [p[1] for p in box]
        x_coords = [p[0] for p in box]
        
        y_center = sum(y_coords) / 4
        area = (max(x_coords) - min(x_coords)) * (max(y_coords) - min(y_coords))

        # Filter: Only bottom 60% of the tag for the main ID
        if y_center > (crop_h * 0.4):
            if area > max_area:
                max_area = area
                largest_text = text

    return clean_and_format(largest_text) if largest_text else None

# --- Main UI ---
st.title("🐄 Advanced Ear Tag OCR")
uploaded_file = st.file_uploader("Upload Herd Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    h, w, _ = img_array.shape
    viz_img = img_array.copy()
    
    results = detector(img_array, conf=0.35) # Lowered conf slightly for distant tags
    detections = results[0].boxes.xyxy.cpu().numpy()
    
    found_tags = []
    
    for box in detections:
        x1, y1, x2, y2 = map(int, box)
        
        # --- BEST BOUNDING BOX OPTION: Dynamic Padding ---
        # Calculate width and height of the initial detection
        bw = x2 - x1
        bh = y2 - y1
        
        # Expand the box by 15% to avoid character clipping at edges
        pad_w = int(bw * 0.15)
        pad_h = int(bh * 0.15)
        
        # Apply expansion while staying within image boundaries
        x1_pad = max(0, x1 - pad_w)
        y1_pad = max(0, y1 - pad_h)
        x2_pad = min(w, x2 + pad_w)
        y2_pad = min(h, y2 + pad_h)
        
        crop = img_array[y1_pad:y2_pad, x1_pad:x2_pad]
        
        if crop.size > 0:
            tag_id = process_tag_ocr(crop)
            display_id = tag_id if tag_id else "???"
            
            # Draw results using OpenCV
            cv2.rectangle(viz_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(viz_img, f"ID: {display_id}", (x1, y1 - 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            found_tags.append({"id": display_id, "crop": crop})
    
    st.image(viz_img, caption="Herd Detection Overview")
    
    if found_tags:
        st.subheader("Individual Tag Details")
        # Use columns to handle multiple cows (e.g., cow1372.jpg)
        cols = st.columns(min(len(found_tags), 5)) 
        for idx, tag in enumerate(found_tags):
            with cols[idx % 5]:
                st.image(tag['crop'])
                st.info(f"Tag ID: **{tag['id']}**")
