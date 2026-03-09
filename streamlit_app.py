import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from rapidocr_onnxruntime import RapidOCR
import re
import os

# --- Configuration ---
st.set_page_config(page_title="🐄 Cattle Eartag detector", layout="wide")

MISHAP_MAP = {
    "|": "1", "I": "1", "l": "1", "[": "1", "]": "1", "(": "1", ")": "1",
    "O": "0", "o": "0", "S": "5", "s": "5", "B": "8", "G": "6"
}

def load_models():
    """Load YOLO detector and RapidOCR recognizer."""
    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, 'cow_eartag_yolov8n_100ep_clean_best.pt')
    if not os.path.exists(model_path):
        st.error("Model file 'cow_eartag_yolov8n_100ep_clean_best.pt' not found.")
        st.stop()
    return YOLO(model_path), RapidOCR()

@st.cache_resource
def get_detector_recognizer():
    """Cache the models to avoid reloading on every rerun."""
    return load_models()

detector, recognizer = get_detector_recognizer()

def clean_and_format(raw_text):
    """Applies mishap mapping and keeps only digits."""
    text = raw_text.strip()
    for char, replacement in MISHAP_MAP.items():
        text = text.replace(char, replacement)
    # Remove any remaining non-digit characters
    return re.sub(r'\D', '', text)

def get_tag_id(crop):
    """Extracts text from the crop and cleans it."""
    result, _ = recognizer(crop)
    if not result:
        return None

    candidates = []
    for line in result:
        text = line[1]
        cleaned = clean_and_format(text)
        if cleaned:
            candidates.append(cleaned)

    # Join multiple lines if the ID is split
    return "".join(candidates) if candidates else None

# --- Main UI ---
st.title("Cattle Ear tag detector")

uploaded_file = st.file_uploader("Upload Tag Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load the image
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    
    # Create a copy specifically for drawing OpenCV bounding boxes
    annotated_img = img_array.copy()
    
    # Run detection
    results = detector(img_array, conf=0.4)
    
    st.subheader("Extracted IDs")
    found_any = False
    tag_crops = []

    # Process all detected boxes
    if results and len(results) > 0:
        for idx, box in enumerate(results[0].boxes.xyxy.cpu().numpy()):
            x1, y1, x2, y2 = map(int, box)
            
            # --- OpenCV: Draw bounding boxes and labels on the main image ---
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(annotated_img, f"Tag {idx+1}", (x1, max(y1 - 10, 20)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Crop the tag
            tag_crop = img_array[y1:y2, x1:x2]
            
            if tag_crop.size > 0:
                # OpenCV Pre-processing to improve dark-on-yellow contrast
                gray = cv2.cvtColor(tag_crop, cv2.COLOR_RGB2GRAY)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                processed = clahe.apply(gray)

                # Extract text using OCR
                extracted_id = get_tag_id(processed)
                
                if extracted_id:
                    st.markdown(f"**Tag {idx+1}:** `{extracted_id}`")
                    tag_crops.append({
                        'id': extracted_id,
                        'crop': tag_crop,
                        'idx': idx + 1
                    })
                    found_any = True

    # Display the annotated image with all detections
    st.subheader("Detected Tags Overview")
    st.image(annotated_img, caption="All detected ear tags highlighted")

    # Display individual crops
    if found_any:
        st.subheader("Individual Tag Crops")
        # Use Streamlit columns to display crops side-by-side (up to 4 across)
        cols = st.columns(min(len(tag_crops), 4) if len(tag_crops) > 0 else 1)
        for i, item in enumerate(tag_crops):
            with cols[i % len(cols)]:
                st.image(item['crop'], caption=f"Tag {item['idx']} - ID: {item['id']}")
                
    if not found_any:
        st.warning("No IDs could be clearly extracted.")
else:
    st.info("Upload an image to begin")
