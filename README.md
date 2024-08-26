# Face Detection -> Age/Gender Classification using ONNX/Hailo Acceleration

This repository provides a Python implementation for real-time face detection and age/gender classification using Hailo AI hardware and the Picamera2 library. The code utilizes ONNX/Hailo models for face detection and classification, processes video input from a camera or a video file, and displays or saves the annotated output.

The code is designed to run efficiently on Hailo AI hardware, making use of its NPU processing capabilities to perform multiple inferences simultaneously.

## Features

- **Real-time Face Detection**: Detects faces in real-time using a pre-trained ONNX/Hailo model.
- **Age and Gender Classification**: Classifies the detected faces into predefined age groups and genders.
- **Hailo AI Hardware Acceleration**: Utilizes Hailo's AI hardware for accelerated inference.
- **Picamera2/OpenCV Integration**: Supports live video feed from Picamera2/OpenCV or video files.
- **Multiprocessing for Parallel Inference**: Uses Python multiprocessing to handle multiple inference tasks concurrently.

## Installation (Hailo with Raspberry Pi 5)

1. **Install/Update Picamera2**

   Refer the installation and updating guide here [picamera2-manual.pdf](https://github.com/user-attachments/files/16744920/picamera2-manual.pdf)

2. **Create vnenv with system packages and Activate**
   ```bash
   mkdir hailo
   python -m venv hailo --system-site-packages
   source hailo/bin/activate
   
3. **Clone the Repository**:
   ```bash
   git clone https://github.com/Arif-Firdaus/FaceDet-onnx.git
   cd FaceDet-onnx
   
5. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt

6. **Install (HailoRT PCIe driver,HailoRT,PyHailoRT) for Ubuntu, Python 3.11**
   1. **Install DKMS**
      ```bash
      sudo apt update && sudo apt install --no-install-recommends dkms
      sudo reboot
   2. **Install (HailoRT PCIe driver,HailoRT,PyHailoRT)**

      Refer the installation guide for HailoRT/ARM64/Linux/Python 3.11 here [hailort_4.18.0_user_guide.pdf](https://github.com/user-attachments/files/16744913/hailort_4.18.0_user_guide.pdf)
      
   4. **Enable multi-process service (to allow running models parallelly)**
      ```bash
      sudo systemctl enable --now hailort.service
      
## Usage...
