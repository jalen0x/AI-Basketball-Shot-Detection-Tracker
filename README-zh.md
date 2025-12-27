# 实时篮球投篮检测（YOLOv8与OpenCV结合使用）
作者：[Avi Shah](https://www.linkedin.com/in/-avishah/) （2023年）

得分检测准确率：95% <br>
投篮检测准确率：97% <br>

https://github.com/avishah3/AI-Basketball-Shot-Detector-Tracker/assets/115107522/469750e1-0f3c-4b07-9fc5-5387831742f7

## 语言选择
- [English](README.md)
- [中文](README-zh.md)

## 项目简介

本项目结合了机器学习和计算机视觉的强大功能，旨在实时检测和分析篮球投篮！基于最新的YOLOv8（You Only Look Once）机器学习模型和OpenCV库，程序可以处理来自各种来源的视频流，如实时摄像头或预录制的视频，为用户提供一种身临其境的游戏体验，并增强比赛分析能力。

## 目录结构

```
├── README.md
├── best.pt                    # PyTorch 训练模型 (6.0 MB)
├── models/                    # 移动端部署模型
│   ├── android/
│   │   └── best_int8.tflite  # 安卓模型 (INT8 量化, 3.2 MB)
│   └── ios/
│       └── best.mlpackage    # iOS 模型 (CoreML FP16, 5.9 MB)
├── config.yaml               # 数据集配置文件
├── main.py                   # YOLOv8 训练脚本
├── pyproject.toml            # 项目依赖配置 (uv/pip)
├── shot_detector.py          # 投篮检测核心程序
├── utils.py                  # 辅助工具文件
├── test_android_model.py     # 安卓模型测试脚本
├── test_ios_model.py         # iOS 模型测试脚本
└── video_test_5.mp4          # 测试视频
```

## 模型训练

训练过程使用了Ultralytics的YOLO实现，并使用了`config.yaml`文件中指定的自定义数据集。模型经过若干训练周期，最终保存下来的最佳模型权重将用于后续的投篮检测。虽然该模型适用于我的使用场景，但不同的数据集或训练方法可能会更适合你的特定项目需求。

## 算法原理

该项目的核心算法使用训练好的YOLOv8模型检测每一帧中的篮球和篮球架。算法分析篮球相对于篮球架的运动和位置，判断是否为成功投篮。

为了提高投篮检测的准确性，算法不仅追踪篮球的位置，还对篮球和篮球架的位置应用数据清洗技术。算法旨在过滤掉不准确的数据点，移除超过某一帧限制的数据，并防止在不同物体之间跳跃，从而保持检测的准确性。

通过线性回归算法预测篮球的轨迹，如果预测的轨迹与篮球架相交，算法就会注册为成功投篮。

## 如何使用此代码

### 安装

```bash
# 克隆仓库
git clone <repository-url>
cd AI-Basketball-Shot-Detection-Tracker

# 使用 uv 安装依赖（推荐）
uv sync

# 或使用 pip
pip install -e .
```

### 模型训练（可选）

1. 下载 `config.yaml` 中指定的数据集，并调整配置文件中的路径以匹配你的本地设置。
2. 按照 `main.py` 中的指示进行模型训练，准备进行投篮检测。

**如果你不想自己训练模型，可以直接使用预训练的 `best.pt` 模型**

### 运行投篮检测

```bash
# 通过摄像头或视频运行实时投篮检测
python shot_detector.py
```

### 移动端部署

`models/` 目录中提供了预导出的移动端模型：

**安卓 (TensorFlow Lite)**
- 模型：`models/android/best_int8.tflite`
- 格式：INT8 量化，优化性能
- 大小：3.2 MB
- 推荐：在设备上使用 NNAPI 或 GPU delegate

**iOS (CoreML)**
- 模型：`models/ios/best.mlpackage`
- 格式：FP16，适配 Apple Neural Engine
- 大小：5.9 MB
- 优化用于 iOS/iPadOS 部署

### 模型测试

测试和对比模型性能：

```bash
# 测试安卓 TFLite 模型 vs PyTorch（并排对比）
python test_android_model.py

# 测试 iOS CoreML 模型 vs PyTorch（并排对比）
python test_ios_model.py
```

**注意：** 在 macOS 上测试移动端模型性能不能反映真实设备性能。TFLite 和 CoreML 模型应在实际的安卓/iOS 设备上测试以获得准确的性能基准。

### 贡献

欢迎对本项目进行贡献 - 提交 Pull Request。对于问题或建议，请在本仓库中提交 Issue。

## 免责声明

模型的性能可能会根据视频源的质量、光照条件以及篮球和篮球架在视频中的清晰度而有所不同。此外，如果视频中有多个篮球和篮球架，程序将无法正常工作。在测试时，输入的视频是在户外拍摄的，使用的是手机摄像头拍摄的地面角度视频。