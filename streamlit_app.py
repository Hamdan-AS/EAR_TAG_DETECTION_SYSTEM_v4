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
    PATH B: Position-based filtering (bottom-50% only)
    
    Optimized for cattle tag format:
    - Top 50%: Small date/batch text (date, lot #, etc) - IGNORED
    - Bottom 50%: Large ID text (main identifier) - TARGET
    
    This ensures we never pick up date/metadata text, only the main ID.
    """
    bgr_crop = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    
    result, _ = recognizer(enhanced)
    if not result: 
        return None
    
    text_blocks = []
    crop_h = enhanced.shape[0]
    
    # Extract OCR blocks from BOTTOM 50% only
    for line in result:
        box, text, conf = line
        yc = sum([p[1] for p in box]) / 4  # Vertical center
        xc = sum([p[0] for p in box]) / 4  # Horizontal center
        
        # STRICT: Only keep text in bottom 50% of crop
        # (where the main ID is guaranteed to be)
        if yc > (crop_h * 0.5):
            text_blocks.append({
                "text": text,
                "xc": xc,
                "conf": conf
            })
    
    if not text_blocks:
        return None
    
    # Sort left-to-right to maintain digit order
    text_blocks.sort(key=lambda x: x["xc"])
    
    # Merge all blocks and clean
    merged = "".join([b["text"] for b in text_blocks])
    return clean_and_format(merged)

# --- Main UI ---
st.title("Cattle Ear Tag Detector & OCR")
st.markdown("**Path B:** Detecting single best tag with position-based (bottom-50%) ID extraction.")

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
    
    if len(detections) == 0:
        st.warning("No tags detected.")
    else:
        # 3. Find the LARGEST detection (by area)
        best_idx = 0
        max_area = 0
        
        for i, box in enumerate(detections):
            x1, y1, x2, y2 = map(int, box)
            area = (x2 - x1) * (y2 - y1)
            if area > max_area:
                max_area = area
                best_idx = i
        
        # 4. Process ONLY the best detection
        box = detections[best_idx]
        x1, y1, x2, y2 = map(int, box)
        
        # --- Bounding Box: 15% Expansion ---
        bw, bh = (x2 - x1), (y2 - y1)
        pad_w, pad_h = int(bw * 0.15), int(bh * 0.15)
        
        x1_pad = max(0, x1 - pad_w)
        y1_pad = max(0, y1 - pad_h)
        x2_pad = min(w, x2 + pad_w)
        y2_pad = min(h, y2 + pad_h)
        
        crop = img_array[y1_pad:y2_pad, x1_pad:x2_pad]
        
        if crop.size == 0:
            st.warning("Could not extract tag crop.")
        else:
            # Run OCR on the expanded crop
            tag_id = process_tag_ocr(crop)
            display_id = tag_id if tag_id else "???"
            
            # 5. OpenCV Visualization - Draw ONLY one box
            cv2.rectangle(viz_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(viz_img, f"ID: {display_id}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # 6. Display Results
            st.subheader("Detection Result")
            st.image(viz_img)
            
            st.subheader("Tag Details")
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(crop, caption="Detected Tag (Crop)")
            
            with col2:
                st.metric("Tag ID", display_id)
                st.info(f"(Largest of {len(detections)} detection(s) found)")
