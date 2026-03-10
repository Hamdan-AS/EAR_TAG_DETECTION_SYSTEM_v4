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
    "|": "1", "I": "1", "l": "1", "[": "1", "]": "1", "(", ")", "1": "1",
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
    """
    Finds text in the bottom half of the crop and selects 
    the largest block by pixel area.
    """
    bgr_crop = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    result, _ = recognizer(enhanced)
    if not result:
        return None

    crop_h, crop_w = enhanced.shape[:2]
    largest_text = ""
    max_area = 0

    for line in result:
        box, text, conf = line
        
        # 1. Calculate bounding box area and vertical center
        # box format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        y_coords = [p[1] for p in box]
        x_coords = [p[0] for p in box]
        
        y_center = sum(y_coords) / 4
        width = max(x_coords) - min(x_coords)
        height = max(y_coords) - min(y_coords)
        area = width * height

        # 2. Heuristic: Only consider text in the bottom 60% of the tag
        # This ignores the smaller dates/numbers printed at the top.
        if y_center > (crop_h * 0.4):
            # 3. Keep the block with the largest area
            if area > max_area:
                max_area = area
                largest_text = text

    return clean_and_format(largest_text) if largest_text else None
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
    
    # 3. Process each detection - all boxes, no size selection
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
