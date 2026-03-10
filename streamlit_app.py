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
    Finds the TALLEST text blocks (Main ID) and combines them left-to-right.
    This prevents the code from losing numbers if OCR splits them into fragments.
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

    for line in result:
        box, text, conf = line
        y_coords = [p[1] for p in box]
        x_coords = [p[0] for p in box]
        
        height = max(y_coords) - min(y_coords)
        x_center = sum(x_coords) / 4
        
        if height > max_height:
            max_height = height
            
        text_blocks.append({"text": text, "height": height, "x": x_center})

    # Keep only the blocks that are at least 60% as tall as the tallest character
    # This automatically ignores small dates/batch numbers.
    main_id_parts = [b for b in text_blocks if b["height"] > (max_height * 0.6)]
    
    # Sort left-to-right to maintain correct number sequence
    main_id_parts.sort(key=lambda x: x["x"])
    
    merged_text = "".join([b["text"] for b in main_id_parts])
    return clean_and_format(merged_text) if merged_text else None

# --- Main UI ---
st.title("🐄 Cattle Ear Tag Detector & OCR")
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    h, w, _ = img_array.shape
    viz_img = img_array.copy()
    
    results = detector(img_array, conf=0.4)
    detections = results[0].boxes.xyxy.cpu().numpy()
    
    if len(detections) == 0:
        st.warning("No tags detected.")
    else:
        # Find Largest Bounding Box (assuming this is the primary cow)
        best_idx = np.argmax([(b[2]-b[0])*(b[3]-b[1]) for b in detections])
        box = detections[best_idx]
        x1, y1, x2, y2 = map(int, box)
        
        # 15% Dynamic Padding for OCR context
        pad_w, pad_h = int((x2-x1)*0.15), int((y2-y1)*0.15)
        x1_p, y1_p = max(0, x1-pad_w), max(0, y1-pad_h)
        x2_p, y2_p = min(w, x2+pad_w), min(h, y2+pad_h)
        
        crop = img_array[y1_p:y2_p, x1_p:x2_p]
        tag_id = process_tag_ocr(crop)
        display_id = tag_id if tag_id else "???"
        
        # Draw Box and Label
        cv2.rectangle(viz_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(viz_img, f"ID: {display_id}", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        st.subheader("Detection Result")
        st.image(viz_img)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(crop, caption="OCR Input (with 15% padding)")
        with col2:
            st.metric("Final Detected ID", display_id)
            st.success("Analysis Complete")
