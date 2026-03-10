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

# FIXED: Corrected syntax error by adding values for "(" and ")" 
MISHAP_MAP = {
    "|": "1", "I": "1", "l": "1", "[": "1", "]": "1", "(": "1", ")": "1",
    "O": "0", "o": "0", "S": "5", "s": "5", "B": "8", "G": "6"
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

    bgr_crop = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    enhanced = cv2.resize(enhanced, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    result, _ = recognizer(enhanced)

    if not result:
        return None

    h = enhanced.shape[0]

    candidates = []

    for line in result:
        box, text, conf = line

        y_coords = [p[1] for p in box]
        x_coords = [p[0] for p in box]

        y_center = sum(y_coords) / 4
        x_center = sum(x_coords) / 4

        # only keep bottom 40% of tag
        if y_center > h * 0.55:
            candidates.append((x_center, text))

    if not candidates:
        return None

    # sort left → right
    candidates = sorted(candidates, key=lambda x: x[0])

    merged = "".join([t[1] for t in candidates])

    cleaned = clean_and_format(merged)

    return cleaned if len(cleaned) >= 4 else None

# --- Main UI ---
st.title("Cattle Ear Tag Detector & OCR")
st.markdown("Extracting only the **Large Bottom ID** from all detected tags.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # 1. Load Image
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    h, w, _ = img_array.shape
    viz_img = img_array.copy()
    
    # 2. Run YOLO Detection
    results = detector(img_array, conf=0.4)
    detections = results[0].boxes.xyxy.cpu().numpy()
    
    found_tags = []
    
    # 3. Process each detection
    for i, box in enumerate(detections):
        x1, y1, x2, y2 = map(int, box)
        
        # --- BEST BOUNDING BOX OPTION: 15% Expansion ---
        # We expand the box slightly to ensure characters on the edge (like '1') aren't cut off
        bw, bh = (x2 - x1), (y2 - y1)
        pad_w, pad_h = int(bw * 0.15), int(bh * 0.15)
        
        x1_pad = max(0, x1 - pad_w)
        y1_pad = max(0, y1 - pad_h)
        x2_pad = min(w, x2 + pad_w)
        y2_pad = min(h, y2 + pad_h)
        
        crop = img_array[y1_pad:y2_pad, x1_pad:x2_pad]
        
        if crop.size == 0:
            continue
            
        # Run OCR on the expanded crop
        tag_id = process_tag_ocr(crop)
        display_id = tag_id if tag_id else "???"
        
        # 4. OpenCV Visualization
        cv2.rectangle(viz_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(viz_img, f"ID: {display_id}", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        found_tags.append({"id": display_id, "crop": crop})
    
    # 5. Display Results
    st.subheader("Detection Result")
    st.image(viz_img)
    
    if found_tags:
        st.subheader("Individual Tag Details")
        cols = st.columns(len(found_tags))
        for idx, tag in enumerate(found_tags):
            with cols[idx]:
                st.image(tag['crop'], caption=f"Tag {idx+1}")
                st.write(f"**ID:** `{tag['id']}`")
    else:
        st.warning("No tags detected.")
