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

MISHAP_MAP = {
    "|": "1", "I": "1", "l": "1", "[": "1", "]": "1", "(": "1", ")": "1",
    "O": "0", "o": "0", "S": "5", "s": "5", "B": "8", "G": "6"
}

@st.cache_resource
def get_models():
    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, 'cow_eartag_yolov8n_100ep_clean_best.pt')
    if not os.path.exists(model_path):
        st.error(f"Model file '{model_path}' not found.")
        st.stop()
    return YOLO(model_path), RapidOCR()

detector, recognizer = get_models()

def clean_and_format(raw_text):
    text = raw_text.strip()
    for char, replacement in MISHAP_MAP.items():
        text = text.replace(char, replacement)
    return re.sub(r'\D', '', text)

def process_tag_ocr(crop):
    """
    Uses HEIGHT-BASED filtering. It identifies the tallest text (the main ID),
    drops all smaller text (the dates/batch numbers), and then sorts the main ID 
    fragments from left-to-right.
    """
    bgr_crop = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    result, _ = recognizer(enhanced)
    if not result:
        return None

    text_blocks = []
    max_height = 0

    # 1. Analyze all detected text blocks
    for line in result:
        box, text, conf = line
        y_coords = [p[1] for p in box]
        x_coords = [p[0] for p in box]
        
        # Calculate height and x-center for this text block
        height = max(y_coords) - min(y_coords)
        x_center = sum(x_coords) / 4
        
        # Track the tallest text block found
        if height > max_height:
            max_height = height
            
        text_blocks.append({
            "text": text,
            "height": height,
            "x_center": x_center
        })

    # 2. Filter out the dates: Keep ONLY blocks that are at least 55% as tall as the tallest text
    main_id_blocks = [b for b in text_blocks if b["height"] > (max_height * 0.55)]
    
    # 3. Sort the surviving large numbers from left-to-right
    main_id_blocks.sort(key=lambda b: b["x_center"])
    
    # 4. Combine and clean
    merged_text = "".join([b["text"] for b in main_id_blocks])
    return clean_and_format(merged_text) if merged_text else None

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
        
        # 15% Padding to ensure edge numbers are visible to OCR
        pad_x = int((x2 - x1) * 0.15)
        pad_y = int((y2 - y1) * 0.15)
        
        x1_p = max(0, x1 - pad_x)
        y1_p = max(0, y1 - pad_y)
        x2_p = min(img_array.shape[1], x2 + pad_x)
        y2_p = min(img_array.shape[0], y2 + pad_y)
        
        crop = img_array[y1_p:y2_p, x1_p:x2_p]
        
        if crop.size == 0:
            continue
            
        tag_id = process_tag_ocr(crop)
        display_id = tag_id if tag_id else "???"
        
        cv2.rectangle(viz_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        label = f"ID: {display_id}"
        cv2.putText(viz_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        found_tags.append({"id": display_id, "crop": crop})
    
    st.subheader("Detection Result")
    st.image(viz_img)
    
    if found_tags:
        st.subheader("Individual Tag Details")
        cols = st.columns(min(len(found_tags), 4))
        for idx, tag in enumerate(found_tags):
            with cols[idx % 4]:
                st.image(tag['crop'])
                st.write(f"**Detected ID:** `{tag['id']}`")
