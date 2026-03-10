import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from rapidocr_onnxruntime import RapidOCR
import re
import os

# --- Configuration ---
st.set_page_config(page_title="Cattle Eartag Detector", layout="wide")

# FIXED: Corrected dictionary syntax with proper string quotes
MISHAP_MAP = {
    "|": "1", "I": "1", "l": "1", "[": "1", "]": "1", "(": "1", ")": "1",
    "O": "0", "o": "0", "S": "5", "s": "5", "B": "8", "G": "6",
    "Q": "0", "D": "0", "Z": "2", "L": "1"
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

def preprocess_for_ocr(crop):
    """
    Pre-processes the crop using CLAHE for better contrast.
    """
    # Convert RGB to BGR for OpenCV
    bgr_crop = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    return enhanced

def process_tag_ocr(crop):
    """
    RULE 2 - Height-Based Filtering: Finds the tallest text block and keeps only
    blocks that are >= 60% of that maximum height to filter out small metadata.
    
    RULE 3 - Horizontal Reassembly: Sorts tall blocks left-to-right and
    concatenates them to handle fragmented IDs.
    """
    # Pre-process with CLAHE
    enhanced = preprocess_for_ocr(crop)
    
    # Run OCR
    result, _ = recognizer(enhanced)
    if not result:
        return None
    
    # Extract height and position for each text block
    blocks = []
    max_height = 0
    
    for line in result:
        box, text, conf = line
        
        # Calculate height of this text block
        y_coords = [p[1] for p in box]
        x_coords = [p[0] for p in box]
        
        height = max(y_coords) - min(y_coords)
        width = max(x_coords) - min(x_coords)
        
        # Calculate horizontal center for sorting
        x_center = sum(x_coords) / 4
        
        # Track maximum height
        if height > max_height:
            max_height = height
        
        blocks.append({
            "text": text,
            "height": height,
            "width": width,
            "x_center": x_center
        })
    
    # RULE 2: Filter to keep only tall blocks (>= 60% of max height)
    # This eliminates small text like dates, batch numbers, manufacturer codes
    min_height_threshold = max_height * 0.6
    tall_blocks = [b for b in blocks if b["height"] >= min_height_threshold]
    
    if not tall_blocks:
        return None
    
    # RULE 3: Sort tall blocks left-to-right for reassembly
    tall_blocks.sort(key=lambda x: x["x_center"])
    
    # Concatenate all tall text blocks to form the full ID
    raw_id = "".join([b["text"] for b in tall_blocks])
    
    return clean_and_format(raw_id)

# --- Main UI ---
st.title("Cattle Ear Tag Detector & OCR")
st.markdown("Using **Height-Based Filtering** to extract only the main ID (tallest text).")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    confidence = st.slider("Detection Confidence", 0.1, 1.0, 0.4, 0.05)
    st.markdown("---")
    st.markdown("**Processing Rules:**")
    st.markdown("1. **Padding**: 15% expansion ensures edge digits are captured")
    st.markdown("2. **Height Filter**: Keeps only text ≥60% of max height")
    st.markdown("3. **Reassembly**: Sorts fragments left-to-right")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # 1. Load Image
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    h, w, _ = img_array.shape
    viz_img = img_array.copy()
    
    # 2. Run YOLO Detection
    results = detector(img_array, conf=confidence)
    detections = results[0].boxes.xyxy.cpu().numpy()
    
    found_tags = []
    
    # 3. Process each detection
    for i, box in enumerate(detections):
        x1, y1, x2, y2 = map(int, box)
        
        # RULE 1 - Padding: Expand bounding box by 15% to capture edge digits
        bw, bh = (x2 - x1), (y2 - y1)
        pad_w, pad_h = int(bw * 0.15), int(bh * 0.15)
        
        x1_pad = max(0, x1 - pad_w)
        y1_pad = max(0, y1 - pad_h)
        x2_pad = min(w, x2 + pad_w)
        y2_pad = min(h, y2 + pad_h)
        
        crop = img_array[y1_pad:y2_pad, x1_pad:x2_pad]
        
        if crop.size == 0:
            continue
        
        # Run OCR with height-based filtering
        tag_id = process_tag_ocr(crop)
        display_id = tag_id if tag_id else "???"
        
        # Visualization
        cv2.rectangle(viz_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(viz_img, f"ID: {display_id}", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        found_tags.append({"id": display_id, "crop": crop})
    
    # 5. Display Results
    st.subheader("Detection Result")
    st.image(viz_img, use_container_width=True)
    
    if found_tags:
        st.subheader("Individual Tag Details")
        cols = st.columns(len(found_tags))
        for idx, tag in enumerate(found_tags):
            with cols[idx]:
                st.image(tag['crop'], caption=f"Tag {idx+1}")
                st.write(f"**ID:** `{tag['id']}`")
    else:
        st.warning("No tags detected.")
