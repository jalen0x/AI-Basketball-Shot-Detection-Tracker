#!/usr/bin/env python3
"""测试 iOS CoreML 模型的准确度 - 实时视频对比"""

from ultralytics import YOLO
import cv2
import numpy as np
import time

def test_ios_model():
    # 加载 iOS CoreML 模型
    print("加载 iOS CoreML 模型...")
    model_coreml = YOLO("models/ios/best.mlpackage", task='detect')

    # 加载原始 PyTorch 模型用于对比
    print("加载原始 PyTorch 模型...")
    model_pt = YOLO("best.pt")

    # 打开测试视频
    cap = cv2.VideoCapture("video_test_5.mp4")

    frame_count = 0
    coreml_times = []
    pytorch_times = []

    print("\n开始实时对比...\n按 'q' 退出\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # 复制帧用于两个模型
        frame_coreml = frame.copy()
        frame_pt = frame.copy()

        # CoreML 模型推理
        start = time.time()
        results_coreml = model_coreml(frame_coreml, verbose=False)
        coreml_time = (time.time() - start) * 1000
        coreml_times.append(coreml_time)

        # 绘制 CoreML 结果
        for r in results_coreml:
            frame_coreml = r.plot()

        # PyTorch 模型推理
        start = time.time()
        results_pt = model_pt(frame_pt, verbose=False)
        pytorch_time = (time.time() - start) * 1000
        pytorch_times.append(pytorch_time)

        # 绘制 PyTorch 结果
        for r in results_pt:
            frame_pt = r.plot()

        # 添加标签和性能信息
        cv2.putText(frame_coreml, f"CoreML FP16", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame_coreml, f"{coreml_time:.1f}ms ({1000/coreml_time:.1f}fps)",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(frame_pt, f"PyTorch FP32", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame_pt, f"{pytorch_time:.1f}ms ({1000/pytorch_time:.1f}fps)",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # 并排显示
        combined = np.hstack((frame_coreml, frame_pt))

        # 缩放以适应屏幕
        scale = 0.8
        width = int(combined.shape[1] * scale)
        height = int(combined.shape[0] * scale)
        combined = cv2.resize(combined, (width, height))

        cv2.imshow('iOS Model Comparison (CoreML vs PyTorch)', combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # 输出统计
    if coreml_times and pytorch_times:
        avg_coreml = np.mean(coreml_times)
        avg_pytorch = np.mean(pytorch_times)

        print("\n" + "="*60)
        print("测试结果:")
        print("="*60)
        print(f"总帧数: {frame_count}")
        print(f"\nCoreML FP16: {avg_coreml:.2f}ms ({1000/avg_coreml:.1f} fps)")
        print(f"PyTorch FP32: {avg_pytorch:.2f}ms ({1000/avg_pytorch:.1f} fps)")
        print(f"速度对比: {avg_pytorch/avg_coreml:.2f}x")
        print("="*60)

if __name__ == "__main__":
    test_ios_model()
