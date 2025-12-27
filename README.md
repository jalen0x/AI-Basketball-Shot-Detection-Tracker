# Real-Time AI Basketball Shot Detection with YOLOv8 and OpenCV
Author: [Avi Shah](https://www.linkedin.com/in/-avishah/) (2023)

Score Detection Accuracy: 95% <br>
Shot Detection Accuracy: 97% <br>

https://github.com/avishah3/AI-Basketball-Shot-Detector-Tracker/assets/115107522/469750e1-0f3c-4b07-9fc5-5387831742f7

## Introduction

This project combines the power of Machine Learning and Computer Vision for the purpose of detecting and analyzing basketball shots in real-time! Built upon the latest YOLOv8 (You Only Look Once) machine learning model and the OpenCV library, the program can process video streams from various sources, such as live webcam feed or pre-recorded videos, providing a tool that can be used for an immersive playing experience and enhanced game analytics.

## Language
- [English](README.md)
- [中文](README-zh.md)

## Directory Structure

```
├── README.md
├── best.pt                    # Pre-trained PyTorch model (6.0 MB)
├── models/                    # Exported models for mobile deployment
│   ├── android/
│   │   └── best_int8.tflite  # Android model (INT8 quantized, 3.2 MB)
│   └── ios/
│       └── best.mlpackage    # iOS model (CoreML FP16, 5.9 MB)
├── config.yaml               # Dataset configuration
├── main.py                   # YOLOv8 training script
├── pyproject.toml            # Project dependencies (uv/pip)
├── shot_detector.py          # Detection algorithm
├── utils.py                  # Helper functions
├── test_android_model.py     # Test script for Android model
├── test_ios_model.py         # Test script for iOS model
└── video_test_5.mp4          # Test video
```

## Model Training

The training process utilizes the ultralytics YOLO implementation and a custom dataset specified in the 'config.yaml' file. The model undergoes a set number of training epochs, with the resulting weights of the best-performing model saved for subsequent usage in shot detection. Although this model worked for my usage, a different dataset or training method might work better for your specific project.

## Algorithm

The core of this project is an algorithm that uses the trained YOLOv8 model to detect basketballs and hoops in each frame. It then analyzes the motion and position of the basketball relative to the hoop to determine if a shot has been made.

To enhance the accuracy of the shot detection, the algorithm not only tracks the ball's position over time but also applies data-cleaning techniques to both the ball and hoop positions. The algorithm is designed to filter out inaccurate data points, remove points beyond a certain frame limit and prevent jumping from one object to another to maintain the accuracy of the detection.

A linear regression is used to predict the ball's trajectory based on its positions. If the projected trajectory intersects with the hoop, the algorithm registers it as a successful shot.

## How to Use This Code

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd AI-Basketball-Shot-Detection-Tracker

# Install dependencies using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

### Training (Optional)

1. Download the dataset specified in 'config.yaml' and adjust the paths in the configuration file to match your local setup.
2. Follow the instructions in 'main.py' to train the model and prepare for shot detection.

**If you don't want to train the model yourself, please use the pre-trained 'best.pt' model**

### Running Shot Detection

```bash
# Run real-time shot detection with webcam or video
python shot_detector.py
```

### Mobile Deployment

Pre-exported models are available in the `models/` directory:

**Android (TensorFlow Lite)**
- Model: `models/android/best_int8.tflite`
- Format: INT8 quantized for optimal performance
- Size: 3.2 MB
- Recommended: Use with NNAPI or GPU delegate on device

**iOS (CoreML)**
- Model: `models/ios/best.mlpackage`
- Format: FP16 for Apple Neural Engine
- Size: 5.9 MB
- Optimized for iOS/iPadOS deployment

### Testing Models

Test and compare model performance:

```bash
# Test Android TFLite model vs PyTorch (side-by-side comparison)
python test_android_model.py

# Test iOS CoreML model vs PyTorch (side-by-side comparison)
python test_ios_model.py
```

**Note:** Mobile model performance testing on macOS does not reflect real device performance. TFLite and CoreML models should be tested on actual Android/iOS devices for accurate benchmarks.

### Contributing

Contributions to this project are welcome - submit a pull request. For issues or suggestions, open an issue in this repository.

## Disclaimer

The model's performance can vary based on factors such as the quality of the video feed, lighting conditions, and the clarity of the basketball and hoop in the video. Furthermore, this program will **not** work if multiple basketballs and hoops are in frame. For testing, this program had input videos that were shot outdoors from a phone camera on the ground.
