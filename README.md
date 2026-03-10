# Cow Ear Tag Detection System v4

An automated computer vision solution for livestock management. This system leverages **YOLOv8** for precise object detection and **RapidOCR** for alphanumeric extraction from cow ear tags.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Installation](#️-installation)


---

## 🔍 Overview

The **Cow Ear Tag Detection System v4** eliminates manual data entry and expensive scanning hardware.

By simply uploading a photo (including mobile formats), users can:

- **Localize**: Detect ear tags using a fine-tuned **YOLOv8n** model  
- **Process**: Automatically crop and preprocess the detected region  
- **Extract**: Convert the visual tag into digital text using **RapidOCR (ONNX)**  

---

## ✨ Key Features

- **Multi-Format Support**  
  Native support for **JPG, PNG, and HEIC/HEIF (iPhone)** images.

- **Edge-Ready Inference**  
  Uses **YOLOv8n (Nano)** for high-speed performance even on modest hardware.

- **Robust OCR**  
  Powered by `rapidocr-onnxruntime` for reliable text extraction without heavy dependencies.

- **Cloud-Native**  
  Optimized for **Streamlit Cloud** with pre-configured `packages.txt` for system-level dependencies.

---

## 🛠️ Installation

### 1. Clone the Repository

```
git clone https://github.com/Hamdan-AS/EAR_TAG_DETECTION_SYSTEM_v4.git
cd EAR_TAG_DETECTION_SYSTEM_v4
```

### 2. System Dependencies

For OpenCV to function correctly in Linux/headless environments, install:
```
sudo apt-get update
sudo apt-get install -y libgl1 libglib2.0-dev
```
### 3. Python Environment

It is recommended to use a virtual environment.
```
pip install -r requirements.txt
```
### 4.Launch the App
```
streamlit run streamlit_app.py
```
