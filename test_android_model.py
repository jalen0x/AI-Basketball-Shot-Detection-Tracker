#!/usr/bin/env python3
"""测试安卓 TFLite 模型的准确度 - 实时视频对比"""

from ultralytics import YOLO
import cv2
import numpy as np
import time

def test_android_model():
    # 加载安卓 INT8 模型
    print("加载安卓 INT8 TFLite 模型...")
    model_tflite = YOLO("models/android/best_int8.tflite", task='detect')

    # 加载原始 PyTorch 模型用于对比
    print("加载原始 PyTorch 模型...")
    model_pt = YOLO("best.pt")

    # 打开测试视频
    cap = cv2.VideoCapture("video_test_5.mp4")

    frame_count = 0
    tflite_times = []
    pytorch_times = []

    print("\n开始实时对比...\n按 'q' 退出\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # 复制帧用于两个模型
        frame_tflite = frame.copy()
        frame_pt = frame.copy()

        # TFLite 模型推理
        start = time.time()
        results_tflite = model_tflite(frame_tflite, verbose=False)
        tflite_time = (time.time() - start) * 1000
        tflite_times.append(tflite_time)

        # 绘制 TFLite 结果
        for r in results_tflite:
            frame_tflite = r.plot()

        # PyTorch 模型推理
        start = time.time()
        results_pt = model_pt(frame_pt, verbose=False)
        pytorch_time = (time.time() - start) * 1000
        pytorch_times.append(pytorch_time)

        # 绘制 PyTorch 结果
        for r in results_pt:
            frame_pt = r.plot()

        # 添加标签和性能信息
        cv2.putText(frame_tflite, f"TFLite INT8", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame_tflite, f"{tflite_time:.1f}ms ({1000/tflite_time:.1f}fps)",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(frame_pt, f"PyTorch FP32", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame_pt, f"{pytorch_time:.1f}ms ({1000/pytorch_time:.1f}fps)",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # 并排显示
        combined = np.hstack((frame_tflite, frame_pt))

        # 缩放以适应屏幕
        scale = 0.8
        width = int(combined.shape[1] * scale)
        height = int(combined.shape[0] * scale)
        combined = cv2.resize(combined, (width, height))

        cv2.imshow('Android Model Comparison (TFLite vs PyTorch)', combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # 输出统计
    if tflite_times and pytorch_times:
        avg_tflite = np.mean(tflite_times)
        avg_pytorch = np.mean(pytorch_times)

        print("\n" + "="*60)
        print("测试结果:")
        print("="*60)
        print(f"总帧数: {frame_count}")
        print(f"\nTFLite INT8: {avg_tflite:.2f}ms ({1000/avg_tflite:.1f} fps)")
        print(f"PyTorch FP32: {avg_pytorch:.2f}ms ({1000/avg_pytorch:.1f} fps)")
        print(f"速度提升: {avg_pytorch/avg_tflite:.2f}x")
        print("="*60)

if __name__ == "__main__":
    test_android_model()
