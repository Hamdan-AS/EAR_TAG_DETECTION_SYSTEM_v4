import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from rapidocr_onnxruntime import RapidOCR
import zipfile
import io
import os

# --- Configuration & Model Loading ---
st.set_page_config(page_title="Cattle Eartag Detection", layout="wide")

@st.cache_resource
def load_models():
    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, 'cow_eartag_yolov8n_100ep_clean_best.pt')
    
    # Simple check to ensure model exists
    if not os.path.exists(model_path):
        st.error(f"Model file 'cow_eartag_yolov8n_100ep_clean_best.pt' not found in {base_path}")
        st.stop()
        
    return YOLO(model_path), RapidOCR()

detector, recognizer = load_models()

def process_image(img_array):
    """Detects tags and runs OCR on a single image array."""
    results = detector(img_array, conf=0.4)
    detections = []
    
    for r in results:
        for box in r.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box)
            # Safe crop
            crop = img_array[y1:y2, x1:x2]
            if crop.size == 0: continue
            
            # Preprocess for RapidOCR (Grayscale + CLAHE)
            gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # OCR Inference
            result, _ = recognizer(enhanced)
            text = " ".join([line[1] for line in result]) if result else "N/A"
            
            detections.append({"box": [x1, y1, x2, y2], "text": text})
    return detections

# --- UI ---
st.title("🐄 Cattle Ear Tag Detector")
uploaded_file = st.file_uploader("Upload Image or ZIP folder", type=["jpg", "jpeg", "png", "zip"])

if uploaded_file:
    # CASE 1: ZIP FILE
    if uploaded_file.name.endswith('.zip'):
        with zipfile.ZipFile(uploaded_file, 'r') as z:
            # Filter for images only, ignoring hidden MacOS folders
            img_list = [f for f in z.namelist() if f.lower().endswith(('.png', '.jpg', '.jpeg')) and not f.startswith('__')]
            
            st.info(f"Processing {len(img_list)} images from ZIP...")
            
            for img_name in img_list:
                with z.open(img_name) as f:
                    img_data = f.read()
                    image = Image.open(io.BytesIO(img_data)).convert("RGB")
                    img_array = np.array(image)
                    
                    # Process
                    dets = process_image(img_array)
                    
                    # Display result for each image in an expander to save space
                    with st.expander(f"Result: {img_name}"):
                        col1, col2 = st.columns(2)
                        col1.image(image, caption=img_name)
                        for d in dets:
                            col2.success(f"Detected ID: **{d['text']}**")

    # CASE 2: SINGLE IMAGE
    else:
        image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image)
        dets = process_image(img_array)
        
        st.image(image)
        for d in dets:
            st.success(f"Detected ID: **{d['text']}**")
