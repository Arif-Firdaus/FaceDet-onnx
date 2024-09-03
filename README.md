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
   ```
   mkdir hailo
   python -m venv hailo --system-site-packages
   source hailo/bin/activate
   ```
3. **Clone the Repository**:
   ```
   git clone https://github.com/Arif-Firdaus/FaceDet-onnx.git
   cd FaceDet-onnx
   ```
4. **Install Python dependencies**
   ```
   pip install -r requirements.txt
   ```
5. **Install (HailoRT PCIe driver,HailoRT,PyHailoRT) for Ubuntu, Python 3.11**
   1. **Install DKMS**
      ```
      sudo apt update && sudo apt install --no-install-recommends dkms
      sudo reboot
      ```
   2. **Install (HailoRT PCIe driver,HailoRT,PyHailoRT)**

      Refer the installation guide for HailoRT/ARM64/Linux/Python 3.11 here [hailort_4.18.0_user_guide.pdf](https://github.com/user-attachments/files/16744913/hailort_4.18.0_user_guide.pdf)
      
   4. **Enable multi-process service (to allow running models parallelly)**
      ```
      sudo systemctl enable --now hailort.service
      ```

## Installation (onnxruntime)

For onnxruntime you can skip the previously mentioned step 1-2 and do steps 3-4 only to start running.
      
## Usage

Run the main script with optional command-line arguments to start face detection and classification.

For onnxruntime:
- `--write-video`: (Optional) Write video stream without and with annotations to the demo folder.
- `--margin`: (Optional) Add a margin to the face bounding box.
```
python main_onnx.py [--write-video] [--margin]
```

For Hailo with RPi5
- `--video-testing`: (Optional) Write video stream with annotations to the demo folder.
- `--margin`: (Optional) Add a margin to the face bounding box.
```
python main_hailo_rpi.py [--video-testing] [--margin]
```

## Model compilation (Custom)

For custom model compilation, refer configs/hailo/resnet18_age.json for creating new configuration file for the custom model.

1. **Docker Script Environment**
```
Will add soon once environment replication with docker is completed
```

2. **Followed by below, to export the custom model to Hailo HEF using the new config**
```
python3 scripts/hailo_export.py configs/hailo/resnet18_age.json
```

3. **Run results will be in the runs/ folder**

## Model compilation notes

Considerations before using Hailo 3.28.0:

1. Model architecture:
   1. No layer normalization: Layer normalization is not supported, batch normalization is preferred
   2. Transpose restriction: refer to the documentation for transpose restrictions
   3. Squeeze & Excitation Kernel Size: Even though it can successfully be translated, the quantization process is sensitive to the S&E kernel size and might fail

2. Practice:
   1. It’s best to run the Hailo export script with a GPU so optimization and quantization can be done
   2. Must use real calibration images ≥ 1024 and not random values

3. Config:
   1. Optimization level
   2. Compression level
   3. calibset_size / dataset_size
   4. compiler_optimization_level
  
## Configuration

Before running the model, ensure you have configured the necessary parameters:

- **Hailo Hardware Architecture**: Set to `hailo8l`.
- **Start Node Names**: Optional, set to `null` if not specified.
- **End Node Names**: Optional, set to `null` if not specified.
- **Images Path**: Path to the calibration images `/calib_data/image1.jpg'.
- **Calibration Data NPY**: Set to `null` to create a new calibration data or set path to a processed calibration data Ex: `calib_data.npy`.
- **Model Name**: Set to model name Ex: `resnet18_age`.
- **Model Path**: Path to the PyTorch model file `models/torch/age.pt`.
- **Batch Size**: Set to `1` for single image inference.
- **Input Resolution**: Set the input image resolution `[224, 224]`.
- **Input Names**: The onnx input tensor name is `["image"]`.
- **Output Names**: The onnx output tensor name is `["age"]`.

### Quantization and Optimization

Include the following optimization configurations in the alls script:

```
normalization1 = normalization([122.770935, 116.74601, 104.093735], [68.500534, 66.63216, 70.323166])
model_optimization_flavor(optimization_level=2, compression_level=1, batch_size=8)
model_optimization_config(compression_params, auto_4bit_weights_ratio=0.2)
model_optimization_config(calibration, batch_size=8, calibset_size=num_calibration_images)
model_optimization_config(globals, multiproc_policy=allowed)
post_quantization_optimization(finetune, policy=enabled, dataset_size=num_calibration_images)
model_optimization_config(checker_cfg, batch_size=4)
```
