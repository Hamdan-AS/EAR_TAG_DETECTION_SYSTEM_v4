# EAR_TAG_DETECTION_SYSTEM_v4
🐄 Cow Ear Tag Detection System v4
A Streamlit web application for automated detection and OCR-based reading of cow ear tags using YOLOv8 object detection and RapidOCR.

📋 Table of Contents

Overview
Features
Live Demo
Installation
Usage
Model
Project Structure
Contributing
License


Overview
The Cow Ear Tag Detection System v4 is a browser-based tool that lets farmers and livestock operators upload photos of cows and automatically:

Detect ear tags using a fine-tuned YOLOv8n model
Crop the detected tag region
Read the tag ID via OCR (RapidOCR)

No scanning guns, no manual data entry — just upload a photo and get the tag number.

Features

📸 Image upload — supports JPG, PNG, and HEIC/HEIF (iPhone photos)
🔍 YOLOv8n detection — fast, lightweight model trained on clean cow ear tag data
🔤 OCR on detected tags — reads alphanumeric tag IDs using RapidOCR (ONNX runtime)
🌐 Browser-based UI — no installation required for end users (via Streamlit)
☁️ Streamlit Cloud ready — includes packages.txt for system-level dependencies


Live Demo

(Add your Streamlit Cloud URL here once deployed)
e.g. https://your-app-name.streamlit.app


Installation
Prerequisites

Python 3.8+
pip

1. Clone the repository
bashgit clone https://github.com/Hamdan-AS/EAR_TAG_DETECTION_SYSTEM_v4.git
cd EAR_TAG_DETECTION_SYSTEM_v4
2. Install system dependencies

These are required for OpenCV to work correctly on Linux/cloud environments.

bashsudo apt-get install -y libgl1 libglib2.0-dev
3. Install Python dependencies
bashpip install -r requirements.txt
4. Run the app
bashstreamlit run streamlit_app.py
Then open your browser at http://localhost:8501.

Usage

Open the app in your browser
Upload a photo of a cow (JPG, PNG, or HEIC from iPhone)
The app will:

Run YOLOv8 detection to locate the ear tag
Draw a bounding box around the detected tag
Run OCR to extract and display the tag ID




Model
PropertyDetailArchitectureYOLOv8n (nano)Training epochs100DatasetCustom clean cow ear tag datasetWeights filecow_eartag_yolov8n_100ep_clean_best.ptFrameworkUltralytics YOLOv8
The model was trained on a curated dataset of cow ear tag images. The "clean" label refers to a filtered dataset with corrected annotations and removed low-quality images.

Project Structure
EAR_TAG_DETECTION_SYSTEM_v4/
│
├── streamlit_app.py                          # Main Streamlit application
├── cow_eartag_yolov8n_100ep_clean_best.pt    # YOLOv8n model weights
├── requirements.txt                          # Python dependencies
├── packages.txt                              # System-level dependencies (for Streamlit Cloud)
└── README.md

Deployment on Streamlit Cloud
This repo is ready for one-click deployment on Streamlit Cloud:

Push the repo to GitHub (including packages.txt and requirements.txt)
Go to share.streamlit.io and connect your repo
Set the main file path to streamlit_app.py
Deploy — Streamlit Cloud will automatically install both system and Python packages


Note: Ensure cow_eartag_yolov8n_100ep_clean_best.pt is included in the repo or loaded from a remote URL in the app code (large files may need Git LFS).


Dependencies
Python (requirements.txt)
PackagePurposestreamlitWeb app frameworkultralyticsYOLOv8 detectionrapidocr-onnxruntimeOCR for reading tag IDspi-heifHEIC/HEIF image support (iPhone photos)opencv-python-headlessImage processing (headless for server use)pillowImage loading and manipulationnumpyNumerical operations
System (packages.txt)
PackagePurposelibgl1OpenCV OpenGL dependencylibglib2.0-devGLib dependency for OpenCV

Contributing
Contributions are welcome!

Fork the repository
Create a feature branch (git checkout -b feature/your-feature)
Commit your changes (git commit -m 'Add some feature')
Push to the branch (git push origin feature/your-feature)
Open a Pull Request
