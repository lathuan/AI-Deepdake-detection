# predict.py
import os
import argparse
import cv2
import matplotlib.pyplot as plt
from utils.model_loader import load_trained_model
from utils.face_detector import FaceDetector
from utils.video_processor import VideoProcessor

def display_result(video_path, result):
    """Hiển thị kết quả dự đoán"""
    color = 'red' if result['prediction'] == 'FAKE' else 'green'
    emoji = '❌' if result['prediction'] == 'FAKE' else '✅'
    
    print(f"\n{'='*60}")
    print(f"{emoji} KẾT QUẢ DỰ ĐOÁN DEEPFAKE {emoji}")
    print(f"{'='*60}")
    print(f"📹 Video: {os.path.basename(video_path)}")
    print(f"🔍 Kết quả: {result['prediction']}")
    print(f"📊 Độ tin cậy: {result['confidence']:.4f}")
    print(f"🎯 Xác suất FAKE: {result['probability']:.4f}")
    print(f"👤 Số khuôn mặt phân tích: {result['num_faces']}")
    
    # Hiển thị khuôn mặt nếu có
    if 'faces_sample' in result and len(result['faces_sample']) > 0:
        print(f"\n🖼️  Mẫu khuôn mặt trích xuất:")
        
        fig, axes = plt.subplots(1, len(result['faces_sample']), figsize=(12, 3))
        if len(result['faces_sample']) == 1:
            axes = [axes]
            
        for i, (face, ax) in enumerate(zip(result['faces_sample'], axes)):
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            ax.imshow(face_rgb)
            ax.set_title(f'Frame {i+1}')
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Deepfake Detection')
    parser.add_argument('--video', type=str, required=True, help='Đường dẫn video cần kiểm tra')
    parser.add_argument('--model', type=str, default='past/best_deepfake_model_dfd.pth', help='Đường dẫn model')
    parser.add_argument('--device', type=str, default='auto', help='Device: auto, cuda, hoặc cpu')
    
    args = parser.parse_args()
    
    # Kiểm tra file video
    if not os.path.exists(args.video):
        print(f"❌ Video không tồn tại: {args.video}")
        return
    
    # Load model
    try:
        model, device = load_trained_model(args.model, args.device)
    except Exception as e:
        print(f"❌ Lỗi khi load model: {e}")
        return
    
    # Khởi tạo face detector và video processor
    face_detector = FaceDetector(model_path='yolov8l-face-lindevs.pt')
    video_processor = VideoProcessor(face_detector)
    
    # Dự đoán
    result = video_processor.predict_video(args.video, model, device)
    
    # Hiển thị kết quả
    display_result(args.video, result)

# Thêm vào cuối predict.py (trước if __name__ == "__main__")

def predict_deepfake(video_path, model_path='model/best_deepfake_model_dfd.pth', device='auto'):
    """
    Hàm dự đoán deepfake cho Flask app
    """
    try:
        # Kiểm tra file video
        if not os.path.exists(video_path):
            return {"error": "Video file not found"}
        
        # Load model
        model, device = load_trained_model(model_path, device)
        
        # Khởi tạo face detector và video processor
        face_detector = FaceDetector(model_path='yolov8l-face-lindevs.pt')
        video_processor = VideoProcessor(face_detector)
        
        # Dự đoán
        result = video_processor.predict_video(video_path, model, device)
        
        return result
        
    except Exception as e:
        return {"error": f"Prediction error: {str(e)}"}

if __name__ == "__main__":
    main()
import os
from utils.model_loader import load_trained_model
from utils.face_detector import FaceDetector
from utils.video_processor import VideoProcessor
import cv2
import base64

def encode_face_to_base64(face):
    _, buffer = cv2.imencode('.jpg', face)
    return base64.b64encode(buffer.tobytes()).decode('utf-8')

def predict_deepfake(video_path, model_path='model/best_deepfake_model_dfd.pth', device='auto'):
    try:
        if not os.path.exists(video_path):
            return {"error": "Video file not found"}

        model, device = load_trained_model(model_path, device)
        face_detector = FaceDetector(model_path='yolov8l-face-lindevs.pt')
        video_processor = VideoProcessor(face_detector)

        result = video_processor.predict_video(video_path, model, device)

        # Encode faces sample sang base64
        if 'faces_sample' in result and len(result['faces_sample']) > 0:
            result['faces_sample'] = [encode_face_to_base64(face) for face in result['faces_sample']]

        return result
    except Exception as e:
        return {"error": f"Prediction error: {str(e)}"}
