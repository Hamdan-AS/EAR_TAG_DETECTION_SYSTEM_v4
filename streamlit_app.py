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

# Mapping common OCR errors for cattle tags
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
    """Uses OpenCV to prep the crop and RapidOCR to find the ID."""
    # Convert RGB to BGR for OpenCV processing
    bgr_crop = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
    
    # Enhance contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Run OCR
    result, _ = recognizer(enhanced)
    if not result:
        return None
    
    # Merge all detected text blocks left-to-right
    text_blocks = []
    for line in result:
        box, text, conf = line
        # Get x-coordinate for sorting
        x_coords = [pt[0] for pt in box]
        x_min = min(x_coords)
        text_blocks.append((x_min, text))
    
    # Sort by x-coordinate (left-to-right)
    text_blocks.sort(key=lambda x: x[0])
    
    # Merge all text
    merged_text = "".join(text for _, text in text_blocks)
    
    return clean_and_format(merged_text)

# --- Main UI ---
st.title("Cattle Ear Tag Detector & OCR")
st.markdown("This version detects **all** visible tags and extracts ID numbers.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # 1. Load Image
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    viz_img = img_array.copy()  # For OpenCV drawing
    
    # 2. Run YOLO Detection
    results = detector(img_array, conf=0.4)
    detections = results[0].boxes.xyxy.cpu().numpy()
    
    found_tags = []
    
    # 3. Process each detection
    for i, box in enumerate(detections):
        x1, y1, x2, y2 = map(int, box)
        
        # Extract crop
        crop = img_array[y1:y2, x1:x2]
        if crop.size == 0:
            continue
            
        # Run OCR on the crop
        tag_id = process_tag_ocr(crop)
        display_id = tag_id if tag_id else "???"
        
        # 4. OpenCV Visualization
        # Draw bounding box
        cv2.rectangle(viz_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        # Draw label background
        label = f"ID: {display_id}"
        cv2.putText(viz_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        found_tags.append({"id": display_id, "crop": crop})
    
    # 5. Display Results
    st.subheader("Detection Result")
    st.image(viz_img, caption="Processed Image with OpenCV Overlays")
    
    if found_tags:
        st.subheader("Individual Tag Details")
        cols = st.columns(len(found_tags))
        for idx, tag in enumerate(found_tags):
            with cols[idx]:
                st.image(tag['crop'], caption=f"Detected ID: {tag['id']}")
                st.write(f"**Tag {idx+1}:** `{tag['id']}`")
    else:
        st.warning("No tags detected.")
else:
    st.info("Please upload an image of cattle to begin detection.")
