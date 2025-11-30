# predict.py (chỉ sửa phần predict_deepfake và create_confidence_timeline)
import uuid
import os
import io
import base64
import argparse
import cv2
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import numpy as np
from datetime import datetime
from utils.model_loader import load_trained_model
from utils.face_detector import FaceDetector
from utils.video_processor import VideoProcessor
from PIL import Image


def create_confidence_timeline(time_confidence_data, overall_prediction=None):
    """Tạo biểu đồ confidence theo thời gian và trả về base64"""
    if not time_confidence_data:
        return None

    plt.figure(figsize=(12, 4))
    
    times = [data['time_sec'] for data in time_confidence_data]
    confidences = [data['confidence'] for data in time_confidence_data]
    
    plt.plot(times, confidences, 'b-', alpha=0.7, linewidth=2, label='Confidence FAKE')
    plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Threshold FAKE/REAL')
    plt.fill_between(times, confidences, 0.5, where=np.array(confidences)>0.5, 
                     alpha=0.3, color='red', label='Suspicious Zone')
    plt.fill_between(times, confidences, 0.5, where=np.array(confidences)<=0.5, 
                     alpha=0.3, color='green', label='Safe Zone')
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Confidence FAKE')
    plt.title('Confidence Timeline')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)

    # Lưu sang base64
    buf = io.BytesIO()
    plt.savefig(buf, format='PNG', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    img_b64 = base64.b64encode(buf.getvalue()).decode()
    return img_b64


def predict_deepfake(video_path, model_path='best_deepfake_model_dfd.pth', device='auto'):
    """Hàm dự đoán để gọi từ Flask hoặc terminal"""
    try:
        if not os.path.exists(video_path):
            return {"error": "Video file not found"}

        model, device = load_trained_model(model_path, device)
        face_detector = FaceDetector()
        video_processor = VideoProcessor(face_detector)

        # Phân tích video chi tiết
        result = video_processor.predict_video_detailed(video_path, model, device)

        # Chuẩn bị frame + heatmap cho web
        frames_for_web = []
        time_confidence_data = []

        if "frame_analysis" in result and result["frame_analysis"]:
            for idx, frame_info in enumerate(result["frame_analysis"]):
                # Lấy ảnh face
                face_bgr = frame_info.get("face_image")
                if face_bgr is not None:
                    face_rgb = face_bgr[..., ::-1]   # BGR -> RGB
                    pil_face = Image.fromarray(face_rgb)
                else:
                    pil_face = None

                # Lấy heatmap
                pil_heatmap = None
                if "heatmap_overlay" in frame_info and frame_info["heatmap_overlay"] is not None:
                    heatmap_rgb = frame_info["heatmap_overlay"][..., ::-1]
                    pil_heatmap = Image.fromarray(heatmap_rgb)

                frames_for_web.append({
                    "frame_index": frame_info.get("frame_index", idx),
                    "confidence": frame_info.get("confidence", 0),
                    "is_suspicious": frame_info.get("is_suspicious", False),
                    "face_image": pil_face,
                    "heatmap_overlay": pil_heatmap
                })

                # Chuẩn bị dữ liệu timeline
                time_confidence_data.append({
                    "time_sec": frame_info.get("time_sec", idx),  # nếu không có time_sec thì dùng idx
                    "confidence": frame_info.get("confidence", 0)
                })

        result["frames_for_web"] = frames_for_web
                # =======================
        # TÍNH REAL / FAKE
        # =======================
        all_conf = [f.get("confidence", 0) for f in result.get("frame_analysis", [])]

        if len(all_conf) > 0:
            avg_conf = float(np.mean(all_conf))  # lấy trung bình
        else:
            avg_conf = 0

        # Ngưỡng 0.5
        # if avg_conf > 0.5:
        #     result["overall_prediction"] = "FAKE"
        # else:
        #     result["overall_prediction"] = "REAL"

        if result['prediction'] == 'FAKE':
            if result['confidence'] > 0.8:
                result["overall_prediction"] = "FAKE"
            elif result['confidence'] > 0.6:
                result["overall_prediction"] = "FAKE"
            else:
                result["overall_prediction"] = "FAKE"
        else:
            if result['confidence'] > 0.8:
                result["overall_prediction"] = "REAL"
            else:
                result["overall_prediction"] = "REAL"


        result["overall_score"] = round(avg_conf, 4)


        # Tạo biểu đồ timeline base64
        result["timeline_base64"] = create_confidence_timeline(time_confidence_data, result.get("overall_prediction"))

        return result

    except Exception as e:
        return {"error": f"Prediction error: {str(e)}"}
