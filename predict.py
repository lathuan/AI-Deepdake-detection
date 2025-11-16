# predict.py
import os
# predict.py
import os
import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from utils.model_loader import load_trained_model
from utils.face_detector import FaceDetector
from utils.video_processor import VideoProcessor

def create_confidence_timeline(time_confidence_data, overall_prediction):
    """Tạo biểu đồ confidence theo thời gian"""
    plt.figure(figsize=(12, 4))
    
    times = [data['time_sec'] for data in time_confidence_data]
    confidences = [data['confidence'] for data in time_confidence_data]
    
    plt.plot(times, confidences, 'b-', alpha=0.7, linewidth=2, label='Confidence FAKE')
    plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Ngưỡng FAKE/REAL')
    plt.fill_between(times, confidences, 0.5, where=np.array(confidences)>0.5, 
                     alpha=0.3, color='red', label='Vùng nghi ngờ FAKE')
    plt.fill_between(times, confidences, 0.5, where=np.array(confidences)<=0.5, 
                     alpha=0.3, color='green', label='Vùng an toàn')
    
    plt.xlabel('Thời gian (giây)')
    plt.ylabel('Confidence FAKE')
    plt.title('BIỂU ĐỒ CONFIDENCE THEO THỜI GIAN')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    
    # Thêm các điểm đánh dấu cho các frame nghi ngờ nhất
    top_suspicious = sorted(zip(times, confidences), key=lambda x: x[1], reverse=True)[:3]
    for time, conf in top_suspicious:
        plt.plot(time, conf, 'ro', markersize=8)
        plt.annotate(f'{conf:.2f}', (time, conf), xytext=(5, 5), 
                    textcoords='offset points', fontweight='bold')
    
    return plt

def display_advanced_result(video_path, result):
    """Hiển thị kết quả phân tích nâng cao"""
    color = 'red' if result['prediction'] == 'FAKE' else 'green'
    emoji = '❌' if result['prediction'] == 'FAKE' else '✅'
    
    print(f"\n{'='*80}")
    print(f"{emoji} PHÂN TÍCH DEEPFAKE CHI TIẾT {emoji}")
    print(f"{'='*80}")
    print(f"📹 Video: {os.path.basename(video_path)}")
    print(f"🎯 KẾT QUẢ: {result['prediction']}")
    print(f"📊 Độ tin cậy tổng: {result['confidence']:.1%}")
    print(f"🔢 Xác suất FAKE: {result['probability']:.4f}")
    print(f"👤 Số frames phân tích: {result['num_faces']}")
    
    # Hiển thị các frame nghi ngờ nhất
    if result['frame_analysis']:
        print(f"\n🔍 {len(result['frame_analysis'])} FRAME NGHI NGỜ NHẤT:")
        
        # Tạo subplot cho các frame nghi ngờ
        num_frames = len(result['frame_analysis'])
        fig, axes = plt.subplots(2, num_frames, figsize=(20, 8))
        
        if num_frames == 1:
            axes = axes.reshape(2, 1)
        
        for i, frame_info in enumerate(result['frame_analysis']):
            # Hàng 1: Ảnh gốc với bounding box
            face_rgb = cv2.cvtColor(frame_info['face_image'], cv2.COLOR_BGR2RGB)
            axes[0, i].imshow(face_rgb)
            axes[0, i].set_title(f'Frame {frame_info["frame_index"]}\nConf: {frame_info["confidence"]:.3f}', 
                               color='red' if frame_info['is_suspicious'] else 'green',
                               fontweight='bold')
            axes[0, i].axis('off')
            
            # Hàng 2: Heatmap overlay
            if 'heatmap_overlay' in frame_info:
                heatmap_rgb = cv2.cvtColor(frame_info['heatmap_overlay'], cv2.COLOR_BGR2RGB)
                axes[1, i].imshow(heatmap_rgb)
                axes[1, i].set_title('Heatmap\n(Vùng AI chú ý)', fontsize=10)
                axes[1, i].axis('off')
            
            # Thêm bounding box màu theo mức độ nghi ngờ
            for spine in axes[0, i].spines.values():
                spine.set_edgecolor('red' if frame_info['is_suspicious'] else 'green')
                spine.set_linewidth(3)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        plt.suptitle('CÁC FRAME NGHI NGỜ NHẤT VÀ HEATMAP PHÂN TÍCH', 
                    fontsize=14, color=color, fontweight='bold')
        plt.show()
    
    # Hiển thị biểu đồ timeline
    if result['time_confidence_data']:
        timeline_plot = create_confidence_timeline(result['time_confidence_data'], result['prediction'])
        timeline_plot.show()
    
    # Thống kê chi tiết
    print(f"\n📈 THỐNG KÊ PHÂN TÍCH:")
    all_confs = result['all_frame_confidences']
    suspicious_frames = sum(1 for conf in all_confs if conf > 0.5)
    avg_confidence = np.mean(all_confs)
    max_confidence = max(all_confs)
    
    print(f"   - Frames nghi ngờ (confidence > 0.5): {suspicious_frames}/{len(all_confs)}")
    print(f"   - Confidence trung bình: {avg_confidence:.3f}")
    print(f"   - Confidence cao nhất: {max_confidence:.3f}")
    print(f"   - Tỉ lệ frames nghi ngờ: {suspicious_frames/len(all_confs):.1%}")
    
    # Phân tích kết luận
    print(f"\n🎯 KẾT LUẬN CHUYÊN SÂU:")
    if result['prediction'] == 'FAKE':
        if result['confidence'] > 0.8:
            print("   🚨 VIDEO CÓ DẤU HIỆU DEEPFAKE RẤT RÕ RÀNG")
            print("   - Nhiều frames có confidence cao")
            print("   - AI phát hiện các bất thường nhất quán")
        elif result['confidence'] > 0.6:
            print("   ⚠️ VIDEO CÓ KHẢ NĂNG CAO LÀ DEEPFAKE")
            print("   - Đa số frames thể hiện dấu hiệu bất thường")
        else:
            print("   🤔 VIDEO NGHI NGỜ DEEPFAKE")
            print("   - Một số frames có dấu hiệu bất thường")
    else:
        if result['confidence'] > 0.8:
            print("   ✅ VIDEO CÓ VẺ HOÀN TOÀN TỰ NHIÊN")
            print("   - Các frames đều thể hiện đặc điểm tự nhiên")
        else:
            print("   👍 VIDEO CÓ KHẢ NĂNG CAO LÀ THẬT")
            print("   - Hầu hết frames không có dấu hiệu bất thường")

def main():
    parser = argparse.ArgumentParser(description='Deepfake Detection - Phiên bản nâng cao')
    parser.add_argument('--video', type=str, required=True, help='Đường dẫn video cần kiểm tra')
    parser.add_argument('--model', type=str, default='best_deepfake_model_dfd.pth', help='Đường dẫn model')
    parser.add_argument('--device', type=str, default='auto', help='Device: auto, cuda, hoặc cpu')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        print(f"❌ Video không tồn tại: {args.video}")
        return
    
    try:
        model, device = load_trained_model(args.model, args.device)
    except Exception as e:
        print(f"❌ Lỗi khi load model: {e}")
        return
    
    face_detector = FaceDetector()
    video_processor = VideoProcessor(face_detector)
    
    # Sử dụng hàm phân tích chi tiết mới
    result = video_processor.predict_video_detailed(args.video, model, device)
    
    # Hiển thị kết quả nâng cao
    display_advanced_result(args.video, result)

if __name__ == "__main__":
    main()

# PostgreSQL connection (Render DATABASE_URL)
DB_URL = os.environ.get("DATABASE_URL")
conn = psycopg2.connect(DB_URL, sslmode="require")

def predict_deepfake(video_path):
    result = process_video_for_api(video_path, model, device)

    # Lưu kết quả vào DB
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO results (filename, prediction, probability) VALUES (%s, %s, %s)",
                (os.path.basename(video_path), result['prediction'], result['probability'])
            )
            conn.commit()
    except Exception as e:
        print(f"❌ Lỗi lưu DB: {e}")

    return result
