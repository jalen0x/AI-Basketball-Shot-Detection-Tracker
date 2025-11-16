# YOLOv8 篮球投篮检测模型 - Flutter 集成指南

## 项目概述

本项目是一个基于 YOLOv8 的篮球投篮检测和追踪模型，用于实时分析篮球投篮动作。

- **模型格式**: PyTorch (.pt)
- **检测类别**: Basketball, Basketball Hoop
- **投篮检测准确率**: 97%
- **得分检测准确率**: 95%

## 模型转换方案

### 为什么需要转换？

Flutter 应用在移动端部署 AI 模型时，推荐使用 TensorFlow Lite 格式：
- 模型体积小（3-10MB）
- 推理速度快
- Flutter 生态支持完善
- 支持 iOS 和 Android 硬件加速

### 转换步骤

#### 1. 环境准备

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate  # Windows

# 安装依赖
pip install ultralytics tensorflow
```

#### 2. 执行转换

**方法一：使用 Python 脚本**

```python
from ultralytics import YOLO

# 加载 PyTorch 模型
model = YOLO("best.pt")

# 转换为 TFLite 格式
model.export(format="tflite")

print("✓ 转换完成!")
print("生成文件: best_saved_model/best_float16.tflite")
```

**方法二：使用命令行**

```bash
yolo export model=best.pt format=tflite
```

#### 3. 转换输出

转换完成后会生成以下文件：

```
best_saved_model/
├── best_float16.tflite    # Float16 量化版本（推荐）
├── best_float32.tflite    # Float32 完整精度版本（可选）
└── saved_model.pb         # TensorFlow SavedModel
```

**推荐使用 `best_float16.tflite`**：
- 精度损失 < 1%
- 模型体积减半
- 推理速度更快

### iOS 特定优化（可选）

如需在 iOS 上获得最佳性能，可转换为 CoreML 格式：

```python
from ultralytics import YOLO

model = YOLO("best.pt")
model.export(format="coreml", nms=True)  # NMS 对检测任务必需
```

## Flutter 集成方案

### 方案对比

| 方案 | 优点 | 缺点 | 推荐度 |
|------|------|------|--------|
| **TFLite + ultralytics_yolo** | 官方支持，易用性高，性能好 | 需要转换模型 | ⭐⭐⭐⭐⭐ |
| **TFLite + flutter_vision** | 支持多种 YOLO 版本 | 文档较少 | ⭐⭐⭐⭐ |
| **PyTorch Mobile + FFI** | 无需转换，直接使用 .pt | 集成复杂，包体积大 | ⭐⭐⭐ |

### 推荐方案：ultralytics_yolo 插件

#### 1. 添加依赖

编辑 `pubspec.yaml`：

```yaml
dependencies:
  flutter:
    sdk: flutter
  ultralytics_yolo: ^0.1.0  # 使用最新版本

flutter:
  assets:
    - assets/models/best_float16.tflite
```

#### 2. 安装插件

```bash
flutter pub get
```

#### 3. 放置模型文件

```
your_flutter_project/
└── assets/
    └── models/
        └── best_float16.tflite
```

#### 4. 初始化模型

```dart
import 'package:ultralytics_yolo/ultralytics_yolo.dart';

class BasketballDetector {
  late ObjectDetector _detector;

  Future<void> initialize() async {
    _detector = ObjectDetector(
      modelPath: 'assets/models/best_float16.tflite',
      modelVersion: 'yolov8',
      numClasses: 2,  // Basketball, Basketball Hoop
      threshold: 0.3,  // 置信度阈值
    );

    await _detector.loadModel();
    print('✓ 模型加载成功');
  }
}
```

#### 5. 执行推理

```dart
import 'dart:io';
import 'package:image_picker/image_picker.dart';

class BasketballDetector {
  // ... 初始化代码 ...

  Future<List<DetectionResult>> detectFromImage(File imageFile) async {
    final results = await _detector.detect(imageFile);
    return results;
  }

  Future<void> detectFromCamera() async {
    final picker = ImagePicker();
    final image = await picker.pickImage(source: ImageSource.camera);

    if (image != null) {
      final results = await detectFromImage(File(image.path));

      for (var result in results) {
        print('类别: ${result.label}');
        print('置信度: ${result.confidence}');
        print('位置: ${result.boundingBox}');
      }
    }
  }
}
```

#### 6. 实时检测

```dart
import 'package:camera/camera.dart';

class RealtimeDetection extends StatefulWidget {
  @override
  _RealtimeDetectionState createState() => _RealtimeDetectionState();
}

class _RealtimeDetectionState extends State<RealtimeDetection> {
  late CameraController _cameraController;
  late ObjectDetector _detector;
  List<DetectionResult> _results = [];

  @override
  void initState() {
    super.initState();
    _initializeCamera();
    _initializeDetector();
  }

  Future<void> _initializeCamera() async {
    final cameras = await availableCameras();
    _cameraController = CameraController(
      cameras[0],
      ResolutionPreset.medium,
    );
    await _cameraController.initialize();

    // 开始实时检测
    _cameraController.startImageStream((CameraImage image) {
      _runDetection(image);
    });
  }

  Future<void> _runDetection(CameraImage image) async {
    final results = await _detector.detectFromCameraImage(image);
    setState(() {
      _results = results;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Stack(
        children: [
          CameraPreview(_cameraController),
          CustomPaint(
            painter: BoundingBoxPainter(_results),
          ),
        ],
      ),
    );
  }
}
```

### 备选方案：flutter_vision 插件

```yaml
dependencies:
  flutter_vision: ^latest_version
```

```dart
import 'package:flutter_vision/flutter_vision.dart';

class BasketballDetector {
  late FlutterVision vision;

  Future<void> initialize() async {
    vision = FlutterVision();
    await vision.loadYoloModel(
      modelPath: 'assets/models/best_float16.tflite',
      modelVersion: 'yolov8',
      numClasses: 2,
    );
  }

  Future<List<Map<String, dynamic>>> detect(File image) async {
    final results = await vision.yoloOnImage(
      bytesList: image.readAsBytesSync(),
      imageHeight: 640,
      imageWidth: 640,
      iouThreshold: 0.4,
      confThreshold: 0.3,
    );
    return results;
  }
}
```

## 性能优化建议

### 1. 模型优化

- **使用 Float16 量化**: 精度损失小，速度提升明显
- **调整输入尺寸**: 默认 640x640，可根据需求调整为 320x320 提速
- **动态批处理**: 对多帧缓存后批量处理

### 2. 平台加速

**Android:**
```dart
// 启用 GPU Delegate
_detector = ObjectDetector(
  modelPath: 'assets/models/best_float16.tflite',
  useGpu: true,  // Android GPU 加速
);
```

**iOS:**
- 使用 CoreML 格式获得最佳性能
- 利用 Neural Engine 硬件加速

### 3. 应用优化

```dart
// 限制推理频率
Timer.periodic(Duration(milliseconds: 100), (timer) {
  // 每 100ms 执行一次检测，避免过度占用 CPU
  _runDetection();
});

// 降采样
final resizedImage = await FlutterImageCompress.compressWithFile(
  imageFile.path,
  minWidth: 640,
  minHeight: 640,
  quality: 85,
);
```

### 4. 包体积优化

```yaml
# 仅打包需要的平台
flutter build apk --split-per-abi  # Android 分平台打包
flutter build ios --split-debug-info  # iOS 精简符号表
```

## 投篮检测算法说明

### 检测逻辑

本模型实现了完整的投篮检测算法：

1. **目标检测**: YOLOv8 检测篮球和篮筐位置
2. **轨迹追踪**: 记录篮球运动轨迹
3. **数据清洗**: 过滤低置信度和异常数据点
4. **线性回归**: 预测篮球飞行路径
5. **进球判定**: 判断轨迹是否穿过篮筐

### 关键参数

```python
# 原始 Python 实现的参数配置
BASKETBALL_CONF_THRESHOLD = 0.3  # 篮球检测置信度
HOOP_CONF_THRESHOLD = 0.5        # 篮筐检测置信度
FRAME_LIMIT = 30                 # 轨迹帧数限制
```

### Flutter 实现参考

需要在 Flutter 中实现对应的投篮检测逻辑：

```dart
class ShotDetectionLogic {
  List<BallPosition> ballPositions = [];
  List<HoopPosition> hoopPositions = [];

  bool _isUp = false;
  bool _isDown = false;
  int makes = 0;
  int attempts = 0;

  void processDetection(List<DetectionResult> results) {
    for (var result in results) {
      if (result.label == 'Basketball' && result.confidence > 0.3) {
        ballPositions.add(BallPosition.fromResult(result));
      }
      if (result.label == 'Basketball Hoop' && result.confidence > 0.5) {
        hoopPositions.add(HoopPosition.fromResult(result));
      }
    }

    _cleanPositions();
    _detectShot();
  }

  void _detectShot() {
    // 实现投篮检测逻辑
    // 参考 shot_detector.py 中的算法
  }
}
```

## 常见问题

### Q1: 转换后精度下降怎么办？

A:
- 使用 Float32 完整精度版本
- 调整转换参数：`model.export(format="tflite", int8=False)`

### Q2: 模型加载失败？

A:
- 检查文件路径是否正确
- 确认 `pubspec.yaml` 中已声明 assets
- 运行 `flutter clean && flutter pub get`

### Q3: 推理速度太慢？

A:
- 启用 GPU 加速
- 降低输入图像分辨率
- 使用 Float16 量化
- 限制推理频率

### Q4: iOS 上无法使用？

A:
- 检查 iOS 最低版本要求（iOS 12+）
- 转换为 CoreML 格式获得更好支持
- 在 `Info.plist` 中添加相机权限

### Q5: 检测结果不准确？

A:
- 调整置信度阈值（threshold）
- 确保光照条件良好
- 模型可能需要针对特定场景重新训练

## 参考资源

- [Ultralytics YOLOv8 文档](https://docs.ultralytics.com/)
- [ultralytics_yolo Flutter 插件](https://pub.dev/packages/ultralytics_yolo)
- [flutter_vision 插件](https://pub.dev/packages/flutter_vision)
- [TensorFlow Lite Flutter](https://www.tensorflow.org/lite/guide/flutter)
- [原项目仓库](https://github.com/avishah3/AI-Basketball-Shot-Detector-Tracker)

## 下一步行动

1. ✅ 环境搭建和依赖安装
2. ✅ 模型转换为 TFLite 格式
3. ⬜ 在 Flutter 项目中集成模型
4. ⬜ 实现投篮检测逻辑
5. ⬜ UI 界面开发
6. ⬜ 性能测试和优化
7. ⬜ 真机测试

---

**作者**: AI-Basketball-Shot-Detection-Tracker
**创建日期**: 2025-11-16
**最后更新**: 2025-11-16
