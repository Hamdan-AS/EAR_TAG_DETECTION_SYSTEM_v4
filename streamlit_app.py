import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from rapidocr_onnxruntime import RapidOCR

# --- Configuration ---
st.set_page_config(page_title="Ear Tag Detection & OCR", layout="wide")

# --- Model Loading (Cached) ---
@st.cache_resource
def load_models():
    # Load YOLOv8 model (ensure cow_eartag_yolov8n_100ep_clean_best.pt is in your repo root)
    detector = YOLO('cow_eartag_yolov8n_100ep_clean_best.pt') 
    # Initialize RapidOCR
    recognizer = RapidOCR()
    return detector, recognizer

detector, recognizer = load_models()

def preprocess_for_ocr(crop):
    """Enhances black text on yellow background."""
    # Convert PIL to OpenCv
    img = cv2.cvtColor(np.array(crop), cv2.COLOR_RGB2BGR)
    
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE to handle lighting variations
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Slight sharpening
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    
    return sharpened

# --- Streamlit UI ---
st.title("🏷️ Ear Tag Recognition System")
st.write("Upload an image to detect livestock tags and extract ID numbers.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load Image
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    # --- Processing ---
    with st.spinner('Detecting tags and reading digits...'):
        # 1. YOLOv8 Detection
        results = detector(img_array, conf=0.4)
        
        detections = []
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            for box in boxes:
                # Crop logic
                x1, y1, x2, y2 = map(int, box)
                crop = image.crop((x1, y1, x2, y2))
                
                # 2. Preprocess Crop
                processed_crop = preprocess_for_ocr(crop)
                
                # 3. RapidOCR Inference
                # RapidOCR accepts numpy arrays (BGR or Gray)
                result, _ = recognizer(processed_crop)
                
                # Extract text if found
                detected_text = ""
                if result:
                    # RapidOCR returns [[box], text, confidence]
                    detected_text = " ".join([line[1] for line in result])
                
                detections.append({
                    "box": [x1, y1, x2, y2],
                    "text": detected_text.strip()
                })

    # --- Results Display ---
    with col2:
        st.subheader("Results")
        
        # Draw boxes and labels on image
        res_img = img_array.copy()
        for det in detections:
            b = det["box"]
            cv2.rectangle(res_img, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 3)
            cv2.putText(res_img, det["text"], (b[0], b[1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            
            st.success(f"Detected Tag ID: **{det['text']}**")

        st.image(res_img, use_container_width=True)

    if not detections:
        st.warning("No ear tags detected in this image.")
