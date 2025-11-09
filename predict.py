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
    parser.add_argument('--model', type=str, default='model/best_deepfake_model_dfd.pth', help='Đường dẫn model')
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

if __name__ == "__main__":
    main()