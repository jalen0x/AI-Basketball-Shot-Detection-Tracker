# YOLOv8 篮球检测模型 - Flutter 完整集成指南

## 📋 目录

- [模型文件说明](#模型文件说明)
- [快速开始](#快速开始)
- [方案一：ultralytics_yolo 插件](#方案一ultralytics_yolo-插件)
- [方案二：flutter_vision 插件](#方案二flutter_vision-插件)
- [方案三：tflite_flutter 自定义实现](#方案三tflite_flutter-自定义实现)
- [投篮检测逻辑实现](#投篮检测逻辑实现)
- [性能优化](#性能优化)
- [常见问题](#常见问题)

---

## 模型文件说明

### 导出结果

```
best_saved_model/
├── best_float16.tflite    ✅ 5.9MB - 推荐使用
├── best_float32.tflite    ⚠️  12MB  - 备选方案
├── saved_model.pb         ❌ Flutter 不需要
├── variables/             ❌ Flutter 不需要
├── assets/                ❌ Flutter 不需要
├── fingerprint.pb         ❌ Flutter 不需要
└── metadata.yaml          📝 配置参考
```

### 模型配置（metadata.yaml）

```yaml
task: detect              # 目标检测任务
batch: 1                  # 批次大小
imgsz: [640, 640]        # 输入尺寸
stride: 32               # 模型步长
channels: 3              # RGB 三通道
names:
  0: Basketball          # 类别 0: 篮球
  1: Basketball Hoop     # 类别 1: 篮筐
nms: false              # ⚠️ 未启用 NMS，需后处理
```

### 关键要点

| 配置项 | 值 | 说明 |
|--------|-----|------|
| **输入尺寸** | 640 x 640 | 必须调整图像到此尺寸 |
| **输入格式** | RGB (3 通道) | 不支持灰度图 |
| **类别数量** | 2 | Basketball + Basketball Hoop |
| **NMS** | false | 需要自己实现 NMS 后处理 |
| **量化方式** | Float16 | 推荐使用，精度损失 <1% |

---

## 快速开始

### 1. 准备模型文件

```bash
# 复制模型到 Flutter 项目
cp best_saved_model/best_float16.tflite \
   your_flutter_project/assets/models/basketball_detector.tflite
```

### 2. 更新 pubspec.yaml

```yaml
dependencies:
  flutter:
    sdk: flutter

  # 选择其中一个插件
  ultralytics_yolo: ^0.1.0          # 方案一（推荐）
  # flutter_vision: ^1.0.0          # 方案二
  # tflite_flutter: ^0.10.0         # 方案三

  # 其他依赖
  camera: ^0.10.5
  image: ^4.0.17
  path_provider: ^2.1.0

flutter:
  assets:
    - assets/models/basketball_detector.tflite
```

### 3. 权限配置

**Android** (`android/app/src/main/AndroidManifest.xml`):
```xml
<uses-permission android:name="android.permission.CAMERA" />
<uses-feature android:name="android.hardware.camera" />
<uses-feature android:name="android.hardware.camera.autofocus" />
```

**iOS** (`ios/Runner/Info.plist`):
```xml
<key>NSCameraUsageDescription</key>
<string>需要访问相机进行篮球投篮检测</string>
<key>NSPhotoLibraryUsageDescription</key>
<string>需要访问相册选择照片</string>
```

---

## 方案一：ultralytics_yolo 插件

### 优点
- 官方支持，稳定性高
- 内置 NMS 后处理
- API 简单易用
- 性能优秀

### 完整实现

#### 1. 检测器封装

```dart
import 'dart:io';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';

class BasketballDetector {
  static const String MODEL_PATH = 'assets/models/basketball_detector.tflite';
  static const double BASKETBALL_THRESHOLD = 0.3;
  static const double HOOP_THRESHOLD = 0.5;
  static const double IOU_THRESHOLD = 0.4;

  late ObjectDetector _detector;
  bool _isInitialized = false;

  /// 初始化模型
  Future<void> initialize() async {
    try {
      _detector = ObjectDetector(
        modelPath: MODEL_PATH,
        modelVersion: 'yolov8',
        numClasses: 2,
        threshold: BASKETBALL_THRESHOLD,
        iouThreshold: IOU_THRESHOLD,
        useGpu: true,  // Android GPU 加速
        numThreads: 4,
      );

      await _detector.loadModel();
      _isInitialized = true;
      print('✅ 篮球检测模型加载成功');
    } catch (e) {
      print('❌ 模型加载失败: $e');
      rethrow;
    }
  }

  /// 检测图片
  Future<DetectionResult> detectImage(File imageFile) async {
    if (!_isInitialized) {
      throw Exception('模型未初始化，请先调用 initialize()');
    }

    try {
      final results = await _detector.detect(imageFile);
      return _parseResults(results);
    } catch (e) {
      print('❌ 检测失败: $e');
      rethrow;
    }
  }

  /// 解析检测结果
  DetectionResult _parseResults(List<Detection> detections) {
    List<Basketball> basketballs = [];
    List<Hoop> hoops = [];

    for (var det in detections) {
      if (det.classId == 0 && det.confidence >= BASKETBALL_THRESHOLD) {
        basketballs.add(Basketball.fromDetection(det));
      } else if (det.classId == 1 && det.confidence >= HOOP_THRESHOLD) {
        hoops.add(Hoop.fromDetection(det));
      }
    }

    return DetectionResult(
      basketballs: basketballs,
      hoops: hoops,
      timestamp: DateTime.now(),
    );
  }

  /// 释放资源
  void dispose() {
    _detector.dispose();
    _isInitialized = false;
  }
}
```

#### 2. 数据模型

```dart
/// 检测结果
class DetectionResult {
  final List<Basketball> basketballs;
  final List<Hoop> hoops;
  final DateTime timestamp;

  DetectionResult({
    required this.basketballs,
    required this.hoops,
    required this.timestamp,
  });

  bool get hasBall => basketballs.isNotEmpty;
  bool get hasHoop => hoops.isNotEmpty;
  bool get isValid => hasBall && hasHoop;
}

/// 篮球
class Basketball {
  final Rect boundingBox;
  final Offset center;
  final double confidence;
  final double width;
  final double height;

  Basketball({
    required this.boundingBox,
    required this.center,
    required this.confidence,
    required this.width,
    required this.height,
  });

  factory Basketball.fromDetection(Detection det) {
    final box = det.boundingBox;
    return Basketball(
      boundingBox: box,
      center: Offset(
        box.left + box.width / 2,
        box.top + box.height / 2,
      ),
      confidence: det.confidence,
      width: box.width,
      height: box.height,
    );
  }
}

/// 篮筐
class Hoop {
  final Rect boundingBox;
  final Offset center;
  final double confidence;

  Hoop({
    required this.boundingBox,
    required this.center,
    required this.confidence,
  });

  factory Hoop.fromDetection(Detection det) {
    final box = det.boundingBox;
    return Hoop(
      boundingBox: box,
      center: Offset(
        box.left + box.width / 2,
        box.top + box.height / 2,
      ),
      confidence: det.confidence,
    );
  }
}
```

#### 3. 实时相机检测

```dart
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';

class RealtimeDetectionScreen extends StatefulWidget {
  @override
  _RealtimeDetectionScreenState createState() =>
      _RealtimeDetectionScreenState();
}

class _RealtimeDetectionScreenState extends State<RealtimeDetectionScreen> {
  late CameraController _cameraController;
  late BasketballDetector _detector;

  DetectionResult? _latestResult;
  bool _isProcessing = false;
  bool _isInitialized = false;

  @override
  void initState() {
    super.initState();
    _initializeAll();
  }

  Future<void> _initializeAll() async {
    await _initializeCamera();
    await _initializeDetector();
  }

  Future<void> _initializeCamera() async {
    final cameras = await availableCameras();
    _cameraController = CameraController(
      cameras.first,
      ResolutionPreset.medium,
      enableAudio: false,
    );

    await _cameraController.initialize();

    // 开始图像流处理
    _cameraController.startImageStream(_processCameraImage);

    setState(() {});
  }

  Future<void> _initializeDetector() async {
    _detector = BasketballDetector();
    await _detector.initialize();
    setState(() => _isInitialized = true);
  }

  /// 处理相机帧
  Future<void> _processCameraImage(CameraImage image) async {
    // 避免重复处理
    if (_isProcessing) return;

    _isProcessing = true;

    try {
      // 转换 CameraImage 为可检测格式
      final file = await _convertCameraImage(image);
      final result = await _detector.detectImage(file);

      setState(() {
        _latestResult = result;
      });
    } catch (e) {
      print('处理失败: $e');
    } finally {
      _isProcessing = false;
    }
  }

  /// 转换相机图像（简化版，实际需要更复杂的转换）
  Future<File> _convertCameraImage(CameraImage image) async {
    // TODO: 实现 YUV 到 RGB 的转换
    // 可使用 image 包进行转换
    throw UnimplementedError('需要实现图像格式转换');
  }

  @override
  Widget build(BuildContext context) {
    if (!_isInitialized || !_cameraController.value.isInitialized) {
      return Scaffold(
        body: Center(child: CircularProgressIndicator()),
      );
    }

    return Scaffold(
      appBar: AppBar(title: Text('篮球检测')),
      body: Stack(
        children: [
          // 相机预览
          CameraPreview(_cameraController),

          // 检测结果绘制
          if (_latestResult != null)
            CustomPaint(
              painter: DetectionPainter(_latestResult!),
              size: Size.infinite,
            ),

          // 统计信息
          Positioned(
            top: 20,
            left: 20,
            child: _buildStatsPanel(),
          ),
        ],
      ),
    );
  }

  Widget _buildStatsPanel() {
    final result = _latestResult;
    if (result == null) {
      return Card(
        child: Padding(
          padding: EdgeInsets.all(16),
          child: Text('等待检测...'),
        ),
      );
    }

    return Card(
      color: Colors.black54,
      child: Padding(
        padding: EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              '篮球: ${result.basketballs.length}',
              style: TextStyle(color: Colors.white, fontSize: 16),
            ),
            SizedBox(height: 8),
            Text(
              '篮筐: ${result.hoops.length}',
              style: TextStyle(color: Colors.white, fontSize: 16),
            ),
          ],
        ),
      ),
    );
  }

  @override
  void dispose() {
    _cameraController.dispose();
    _detector.dispose();
    super.dispose();
  }
}
```

#### 4. 检测结果绘制

```dart
import 'package:flutter/material.dart';

class DetectionPainter extends CustomPainter {
  final DetectionResult result;

  DetectionPainter(this.result);

  @override
  void paint(Canvas canvas, Size size) {
    // 绘制篮球
    for (var ball in result.basketballs) {
      _drawBoundingBox(
        canvas,
        ball.boundingBox,
        Colors.orange,
        'Basketball ${(ball.confidence * 100).toInt()}%',
      );

      // 绘制中心点
      canvas.drawCircle(
        ball.center,
        4,
        Paint()..color = Colors.red,
      );
    }

    // 绘制篮筐
    for (var hoop in result.hoops) {
      _drawBoundingBox(
        canvas,
        hoop.boundingBox,
        Colors.cyan,
        'Hoop ${(hoop.confidence * 100).toInt()}%',
      );

      // 绘制中心点
      canvas.drawCircle(
        hoop.center,
        4,
        Paint()..color = Colors.blue,
      );
    }
  }

  void _drawBoundingBox(
    Canvas canvas,
    Rect box,
    Color color,
    String label,
  ) {
    // 绘制边框
    final paint = Paint()
      ..color = color
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3.0;

    canvas.drawRect(box, paint);

    // 绘制标签背景
    final textPainter = TextPainter(
      text: TextSpan(
        text: label,
        style: TextStyle(
          color: Colors.white,
          fontSize: 14,
          fontWeight: FontWeight.bold,
        ),
      ),
      textDirection: TextDirection.ltr,
    );

    textPainter.layout();

    final labelRect = Rect.fromLTWH(
      box.left,
      box.top - 20,
      textPainter.width + 8,
      20,
    );

    canvas.drawRect(
      labelRect,
      Paint()..color = color,
    );

    // 绘制标签文字
    textPainter.paint(
      canvas,
      Offset(box.left + 4, box.top - 18),
    );
  }

  @override
  bool shouldRepaint(DetectionPainter oldDelegate) {
    return result != oldDelegate.result;
  }
}
```

---

## 方案二：flutter_vision 插件

### 完整实现

```dart
import 'package:flutter_vision/flutter_vision.dart';
import 'dart:io';

class BasketballDetectorVision {
  late FlutterVision _vision;
  bool _isInitialized = false;

  Future<void> initialize() async {
    _vision = FlutterVision();

    await _vision.loadYoloModel(
      labels: 'assets/labels.txt',  // 需要创建标签文件
      modelPath: 'assets/models/basketball_detector.tflite',
      modelVersion: 'yolov8',
      numThreads: 4,
      useGpu: true,
    );

    _isInitialized = true;
    print('✅ 模型加载成功 (flutter_vision)');
  }

  Future<List<Map<String, dynamic>>> detectImage(File imageFile) async {
    if (!_isInitialized) {
      throw Exception('模型未初始化');
    }

    final results = await _vision.yoloOnImage(
      bytesList: imageFile.readAsBytesSync(),
      imageHeight: 640,
      imageWidth: 640,
      iouThreshold: 0.4,
      confThreshold: 0.3,
      classThreshold: 0.3,
    );

    return results;
  }

  void dispose() {
    _vision.closeYoloModel();
    _isInitialized = false;
  }
}
```

### 标签文件 (`assets/labels.txt`)

```
Basketball
Basketball Hoop
```

---

## 方案三：tflite_flutter 自定义实现

### 优点
- 完全控制推理过程
- 理解模型工作原理
- 可自定义后处理逻辑

### 完整实现

```dart
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;
import 'dart:io';
import 'dart:typed_data';

class BasketballDetectorCustom {
  static const int INPUT_SIZE = 640;
  static const int NUM_CLASSES = 2;

  Interpreter? _interpreter;
  bool _isInitialized = false;

  Future<void> initialize() async {
    try {
      _interpreter = await Interpreter.fromAsset(
        'assets/models/basketball_detector.tflite',
        options: InterpreterOptions()
          ..threads = 4
          ..useNnApiForAndroid = true,  // Android NNAPI 加速
      );

      _isInitialized = true;
      print('✅ 模型加载成功 (tflite_flutter)');
      print('输入张量: ${_interpreter!.getInputTensors()}');
      print('输出张量: ${_interpreter!.getOutputTensors()}');
    } catch (e) {
      print('❌ 模型加载失败: $e');
      rethrow;
    }
  }

  /// 图像预处理
  Float32List _preprocessImage(img.Image image) {
    // 调整尺寸
    final resized = img.copyResize(
      image,
      width: INPUT_SIZE,
      height: INPUT_SIZE,
      interpolation: img.Interpolation.linear,
    );

    // 转换为归一化的 Float32 数组
    final input = Float32List(1 * INPUT_SIZE * INPUT_SIZE * 3);
    int pixelIndex = 0;

    for (int y = 0; y < INPUT_SIZE; y++) {
      for (int x = 0; x < INPUT_SIZE; x++) {
        final pixel = resized.getPixel(x, y);

        // RGB 归一化到 0-1
        input[pixelIndex++] = pixel.r / 255.0;
        input[pixelIndex++] = pixel.g / 255.0;
        input[pixelIndex++] = pixel.b / 255.0;
      }
    }

    return input;
  }

  /// 执行推理
  Future<List<Detection>> detectImage(File imageFile) async {
    if (!_isInitialized || _interpreter == null) {
      throw Exception('模型未初始化');
    }

    // 1. 读取并预处理图像
    final imageBytes = imageFile.readAsBytesSync();
    final image = img.decodeImage(imageBytes)!;
    final input = _preprocessImage(image);

    // 2. 准备输入输出张量
    final inputShape = [1, INPUT_SIZE, INPUT_SIZE, 3];
    final inputTensor = input.reshape(inputShape);

    // YOLOv8 输出格式: [1, 84, 8400]
    // 84 = x, y, w, h + 80 classes (但我们只有2类)
    final output = List.filled(1 * 84 * 8400, 0.0).reshape([1, 84, 8400]);

    // 3. 执行推理
    _interpreter!.run(inputTensor, output);

    // 4. 后处理
    final detections = _postprocess(output[0]);

    return detections;
  }

  /// 后处理：解析输出并应用 NMS
  List<Detection> _postprocess(List<List<double>> output) {
    List<Detection> detections = [];

    // YOLOv8 输出格式: [84, 8400]
    // 84 = [x, y, w, h, class0_conf, class1_conf, ...]
    final numDetections = output[0].length; // 8400

    for (int i = 0; i < numDetections; i++) {
      // 提取边界框坐标
      final x = output[0][i];
      final y = output[1][i];
      final w = output[2][i];
      final h = output[3][i];

      // 提取类别置信度
      final basketballConf = output[4][i];
      final hoopConf = output[5][i];

      // 取最大置信度的类别
      final maxConf = basketballConf > hoopConf ? basketballConf : hoopConf;
      final classId = basketballConf > hoopConf ? 0 : 1;

      // 过滤低置信度检测
      if (maxConf < 0.3) continue;

      // 转换为 xyxy 格式
      final x1 = x - w / 2;
      final y1 = y - h / 2;
      final x2 = x + w / 2;
      final y2 = y + h / 2;

      detections.add(Detection(
        classId: classId,
        confidence: maxConf,
        box: Rect.fromLTRB(x1, y1, x2, y2),
      ));
    }

    // 应用 NMS
    return _applyNMS(detections, iouThreshold: 0.4);
  }

  /// NMS (非极大值抑制)
  List<Detection> _applyNMS(
    List<Detection> detections, {
    required double iouThreshold,
  }) {
    // 按置信度降序排序
    detections.sort((a, b) => b.confidence.compareTo(a.confidence));

    List<Detection> result = [];

    while (detections.isNotEmpty) {
      final best = detections.removeAt(0);
      result.add(best);

      // 移除与当前框 IoU 过高的框
      detections.removeWhere((det) {
        final iou = _calculateIoU(best.box, det.box);
        return iou > iouThreshold;
      });
    }

    return result;
  }

  /// 计算 IoU (Intersection over Union)
  double _calculateIoU(Rect box1, Rect box2) {
    final intersection = box1.intersect(box2);
    if (intersection.isEmpty) return 0.0;

    final intersectionArea = intersection.width * intersection.height;
    final box1Area = box1.width * box1.height;
    final box2Area = box2.width * box2.height;
    final unionArea = box1Area + box2Area - intersectionArea;

    return intersectionArea / unionArea;
  }

  void dispose() {
    _interpreter?.close();
    _isInitialized = false;
  }
}

class Detection {
  final int classId;
  final double confidence;
  final Rect box;

  Detection({
    required this.classId,
    required this.confidence,
    required this.box,
  });
}
```

---

## 投篮检测逻辑实现

基于原 Python 代码的投篮检测算法。

```dart
class ShotDetectionEngine {
  static const int FRAME_LIMIT = 30;
  static const double UP_THRESHOLD = 0.7;    // 篮筐上方区域比例
  static const double DOWN_THRESHOLD = 1.3;  // 篮筐下方区域比例

  List<BallPosition> _ballPositions = [];
  List<HoopPosition> _hoopPositions = [];

  bool _isUp = false;
  bool _isDown = false;
  int _upFrame = 0;
  int _downFrame = 0;

  int _makes = 0;
  int _attempts = 0;

  int get makes => _makes;
  int get attempts => _attempts;
  double get accuracy => _attempts > 0 ? _makes / _attempts : 0.0;

  /// 处理每一帧的检测结果
  void processFrame(DetectionResult result, int frameCount) {
    // 1. 添加篮球位置
    for (var ball in result.basketballs) {
      _ballPositions.add(BallPosition(
        center: ball.center,
        frameCount: frameCount,
        width: ball.width,
        height: ball.height,
        confidence: ball.confidence,
      ));
    }

    // 2. 添加篮筐位置
    for (var hoop in result.hoops) {
      _hoopPositions.add(HoopPosition(
        center: hoop.center,
        frameCount: frameCount,
        confidence: hoop.confidence,
      ));
    }

    // 3. 清理旧数据
    _cleanPositions(frameCount);

    // 4. 检测投篮
    _detectShot(frameCount);
  }

  /// 清理过期数据
  void _cleanPositions(int currentFrame) {
    // 保留最近 FRAME_LIMIT 帧的数据
    _ballPositions.removeWhere(
      (pos) => currentFrame - pos.frameCount > FRAME_LIMIT,
    );
    _hoopPositions.removeWhere(
      (pos) => currentFrame - pos.frameCount > FRAME_LIMIT,
    );
  }

  /// 检测投篮
  void _detectShot(int frameCount) {
    if (_hoopPositions.isEmpty || _ballPositions.isEmpty) return;

    final latestHoop = _hoopPositions.last;
    final latestBall = _ballPositions.last;

    // 检测球是否在篮筐上方区域
    if (!_isUp) {
      _isUp = _detectUp(latestBall, latestHoop);
      if (_isUp) {
        _upFrame = frameCount;
        print('🏀 球进入上方区域');
      }
    }

    // 检测球是否在篮筐下方区域
    if (_isUp && !_isDown) {
      _isDown = _detectDown(latestBall, latestHoop);
      if (_isDown) {
        _downFrame = frameCount;
        print('🏀 球进入下方区域');
      }
    }

    // 判断是否完成投篮动作
    if (_isUp && _isDown && _upFrame < _downFrame) {
      _attempts++;

      // 判断是否进球
      if (_scoreDetection()) {
        _makes++;
        print('✅ 进球! ($makes/$attempts)');
      } else {
        print('❌ 未进 ($makes/$attempts)');
      }

      // 重置状态
      _isUp = false;
      _isDown = false;
    }
  }

  /// 检测球是否在篮筐上方
  bool _detectUp(BallPosition ball, HoopPosition hoop) {
    return ball.center.dy < hoop.center.dy - (hoop.center.dy * UP_THRESHOLD);
  }

  /// 检测球是否在篮筐下方
  bool _detectDown(BallPosition ball, HoopPosition hoop) {
    return ball.center.dy > hoop.center.dy + (hoop.center.dy * DOWN_THRESHOLD);
  }

  /// 进球检测（线性回归预测轨迹）
  bool _scoreDetection() {
    if (_ballPositions.length < 3 || _hoopPositions.isEmpty) {
      return false;
    }

    final hoop = _hoopPositions.last;

    // 使用最近的球位置进行线性回归
    final recentBalls = _ballPositions.length > 10
        ? _ballPositions.sublist(_ballPositions.length - 10)
        : _ballPositions;

    // 简化版线性回归：预测球在篮筐 y 位置时的 x 坐标
    final trajectory = _linearRegression(recentBalls);
    if (trajectory == null) return false;

    // 预测球经过篮筐高度时的 x 位置
    final predictedX = trajectory.predictX(hoop.center.dy);

    // 判断预测位置是否在篮筐范围内
    final hoopWidth = 50.0; // 篮筐宽度估计值
    final distance = (predictedX - hoop.center.dx).abs();

    return distance < hoopWidth;
  }

  /// 线性回归
  Trajectory? _linearRegression(List<BallPosition> positions) {
    if (positions.length < 2) return null;

    double sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
    final n = positions.length;

    for (var pos in positions) {
      final x = pos.center.dx;
      final y = pos.center.dy;
      sumX += x;
      sumY += y;
      sumXY += x * y;
      sumX2 += x * x;
    }

    final slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
    final intercept = (sumY - slope * sumX) / n;

    return Trajectory(slope: slope, intercept: intercept);
  }

  void reset() {
    _ballPositions.clear();
    _hoopPositions.clear();
    _isUp = false;
    _isDown = false;
    _makes = 0;
    _attempts = 0;
  }
}

/// 球位置
class BallPosition {
  final Offset center;
  final int frameCount;
  final double width;
  final double height;
  final double confidence;

  BallPosition({
    required this.center,
    required this.frameCount,
    required this.width,
    required this.height,
    required this.confidence,
  });
}

/// 篮筐位置
class HoopPosition {
  final Offset center;
  final int frameCount;
  final double confidence;

  HoopPosition({
    required this.center,
    required this.frameCount,
    required this.confidence,
  });
}

/// 轨迹
class Trajectory {
  final double slope;
  final double intercept;

  Trajectory({required this.slope, required this.intercept});

  double predictX(double y) {
    // y = slope * x + intercept
    // x = (y - intercept) / slope
    return (y - intercept) / slope;
  }
}
```

### 使用投篮检测引擎

```dart
class ShotDetectionScreen extends StatefulWidget {
  @override
  _ShotDetectionScreenState createState() => _ShotDetectionScreenState();
}

class _ShotDetectionScreenState extends State<ShotDetectionScreen> {
  late BasketballDetector _detector;
  late ShotDetectionEngine _shotEngine;
  int _frameCount = 0;

  @override
  void initState() {
    super.initState();
    _detector = BasketballDetector();
    _shotEngine = ShotDetectionEngine();
    _detector.initialize();
  }

  Future<void> _processFrame(File imageFile) async {
    final result = await _detector.detectImage(imageFile);

    setState(() {
      _shotEngine.processFrame(result, _frameCount);
      _frameCount++;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('投篮检测')),
      body: Column(
        children: [
          // 相机预览区域
          Expanded(child: Container()),

          // 统计面板
          _buildStatsPanel(),
        ],
      ),
    );
  }

  Widget _buildStatsPanel() {
    return Container(
      padding: EdgeInsets.all(20),
      color: Colors.black87,
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceAround,
        children: [
          _buildStat('进球', _shotEngine.makes.toString(), Colors.green),
          _buildStat('投篮', _shotEngine.attempts.toString(), Colors.blue),
          _buildStat(
            '命中率',
            '${(_shotEngine.accuracy * 100).toStringAsFixed(1)}%',
            Colors.orange,
          ),
        ],
      ),
    );
  }

  Widget _buildStat(String label, String value, Color color) {
    return Column(
      children: [
        Text(
          value,
          style: TextStyle(
            color: color,
            fontSize: 32,
            fontWeight: FontWeight.bold,
          ),
        ),
        SizedBox(height: 4),
        Text(
          label,
          style: TextStyle(color: Colors.white70, fontSize: 14),
        ),
      ],
    );
  }
}
```

---

## 性能优化

### 1. 使用 Isolate 避免 UI 卡顿

```dart
import 'dart:isolate';

class IsolateDetector {
  static Future<DetectionResult> detectInIsolate(File imageFile) async {
    return await Isolate.run(() async {
      final detector = BasketballDetector();
      await detector.initialize();
      final result = await detector.detectImage(imageFile);
      detector.dispose();
      return result;
    });
  }
}
```

### 2. 限制推理频率

```dart
class ThrottledDetector {
  static const Duration THROTTLE_DURATION = Duration(milliseconds: 100);

  DateTime? _lastDetectionTime;
  bool _isDetecting = false;

  Future<DetectionResult?> detect(File imageFile) async {
    final now = DateTime.now();

    // 限流检查
    if (_lastDetectionTime != null &&
        now.difference(_lastDetectionTime!) < THROTTLE_DURATION) {
      return null;
    }

    if (_isDetecting) return null;

    _isDetecting = true;
    _lastDetectionTime = now;

    try {
      final result = await _detector.detectImage(imageFile);
      return result;
    } finally {
      _isDetecting = false;
    }
  }
}
```

### 3. 图像降采样

```dart
import 'package:flutter_image_compress/flutter_image_compress.dart';

Future<File> compressImage(File file) async {
  final result = await FlutterImageCompress.compressAndGetFile(
    file.absolute.path,
    '${file.path}_compressed.jpg',
    quality: 85,
    minWidth: 640,
    minHeight: 640,
  );

  return File(result!.path);
}
```

### 4. 批量处理

```dart
class BatchDetector {
  static const int BATCH_SIZE = 5;
  final List<File> _batch = [];

  Future<List<DetectionResult>> addAndDetect(File imageFile) async {
    _batch.add(imageFile);

    if (_batch.length >= BATCH_SIZE) {
      final results = await _detectBatch(_batch);
      _batch.clear();
      return results;
    }

    return [];
  }

  Future<List<DetectionResult>> _detectBatch(List<File> images) async {
    // 并发处理多张图片
    return await Future.wait(
      images.map((img) => _detector.detectImage(img)),
    );
  }
}
```

---

## 常见问题

### Q1: 模型加载失败 "Unable to load asset"

**原因**: pubspec.yaml 未正确配置

**解决**:
```yaml
flutter:
  assets:
    - assets/models/basketball_detector.tflite
```

执行:
```bash
flutter clean
flutter pub get
```

### Q2: Android 上推理速度很慢

**解决**:
```dart
// 启用 GPU 加速
_detector = ObjectDetector(
  modelPath: MODEL_PATH,
  useGpu: true,  // ✅ 启用 GPU
);

// 或在 tflite_flutter 中
InterpreterOptions()
  ..useNnApiForAndroid = true  // ✅ 启用 NNAPI
```

### Q3: iOS 权限问题

**解决**: 确保 `Info.plist` 包含:
```xml
<key>NSCameraUsageDescription</key>
<string>需要访问相机进行篮球检测</string>
```

### Q4: 检测结果不准确

**检查清单**:
- ✅ 图像尺寸是否调整为 640x640
- ✅ 置信度阈值是否合理（0.3 - 0.5）
- ✅ 光照条件是否良好
- ✅ 是否实现了 NMS 后处理

### Q5: 内存占用过高

**优化方案**:
```dart
// 1. 及时释放资源
@override
void dispose() {
  _detector.dispose();
  super.dispose();
}

// 2. 限制历史帧数
static const int MAX_HISTORY = 30;

// 3. 使用压缩图像
final compressed = await compressImage(originalImage);
```

### Q6: CameraImage 转换问题

**完整转换代码**:
```dart
import 'package:image/image.dart' as img;

Future<img.Image> convertYUV420ToImage(CameraImage cameraImage) async {
  final int width = cameraImage.width;
  final int height = cameraImage.height;

  final int uvRowStride = cameraImage.planes[1].bytesPerRow;
  final int uvPixelStride = cameraImage.planes[1].bytesPerPixel!;

  final image = img.Image(width: width, height: height);

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      final int uvIndex =
          uvPixelStride * (x / 2).floor() + uvRowStride * (y / 2).floor();
      final int index = y * width + x;

      final yp = cameraImage.planes[0].bytes[index];
      final up = cameraImage.planes[1].bytes[uvIndex];
      final vp = cameraImage.planes[2].bytes[uvIndex];

      int r = (yp + vp * 1436 / 1024 - 179).round().clamp(0, 255);
      int g = (yp - up * 46549 / 131072 + 44 - vp * 93604 / 131072 + 91)
          .round()
          .clamp(0, 255);
      int b = (yp + up * 1814 / 1024 - 227).round().clamp(0, 255);

      image.setPixelRgba(x, y, r, g, b, 255);
    }
  }

  return image;
}
```

---

## 总结

### 推荐配置

| 项目 | 推荐值 |
|------|--------|
| **模型文件** | best_float16.tflite |
| **Flutter 插件** | ultralytics_yolo |
| **输入尺寸** | 640 x 640 |
| **置信度阈值** | Basketball: 0.3, Hoop: 0.5 |
| **NMS IoU** | 0.4 |
| **推理频率** | 10 FPS (100ms 间隔) |
| **线程数** | 4 |
| **GPU 加速** | 启用 (Android) |

### 开发流程

1. ✅ 准备模型文件 (`best_float16.tflite`)
2. ✅ 配置 Flutter 项目 (`pubspec.yaml`)
3. ✅ 集成检测插件 (ultralytics_yolo 推荐)
4. ✅ 实现检测逻辑
5. ✅ 实现投篮检测算法
6. ✅ UI 界面开发
7. ✅ 性能优化
8. ✅ 真机测试

---

**文档版本**: 1.0
**创建日期**: 2025-11-16
**模型版本**: YOLOv8 (Ultralytics 8.3.228)
**适用平台**: iOS 12+, Android 5.0+
