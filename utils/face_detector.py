# utils/face_detector.py
from ultralytics import YOLO
import cv2
import os

class FaceDetector:
    def __init__(self, model_path='yolov8l-face-lindevs.pt'): 
        """
        Khởi tạo face detector với YOLO
        
        Args:
            model_path: Đường dẫn đến model YOLO custom
        """
        # Kiểm tra xem file model có tồn tại không
        if not os.path.exists(model_path):
            print(f"⚠️ Không tìm thấy model: {model_path}")
            print("📥 Đang tải model mặc định...")
            model_path = 'yolov8l-face-lindevs.pt'
        
        try:
            self.model = YOLO(model_path)
            print(f"✅ Đã load face detection model: {os.path.basename(model_path)}")
        except Exception as e:
            print(f"❌ Lỗi khi load model {model_path}: {e}")
            print("🔄 Đang thử load model mặc định...")
            self.model = YOLO('yolov8l-face-lindevs.pt')
        
    def extract_faces_from_video(self, video_path, max_frames=20, conf_threshold=0.7):
        """
        Trích xuất khuôn mặt từ video
        
        Args:
            video_path: Đường dẫn video
            max_frames: Số frame tối đa
            conf_threshold: Ngưỡng confidence
            
        Returns:
            List các khuôn mặt đã cắt
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"❌ Không thể mở video: {video_path}")
                return []
            
            face_frames = []
            frame_count = 0
            max_frames_to_process = 100
            
            while len(face_frames) < max_frames and frame_count < max_frames_to_process:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_count += 1
                
                # Xử lý mỗi 5 frame để tiết kiệm thời gian
                if frame_count % 5 != 0:
                    continue
                
                # Chuyển BGR sang RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Phát hiện khuôn mặt với YOLO
                results = self.model(frame_rgb, conf=conf_threshold, verbose=False)
                
                for result in results:
                    if len(result.boxes) > 0:
                        for box in result.boxes:
                            confidence = box.conf[0].item()
                            if confidence > conf_threshold:
                                # Lấy tọa độ bounding box
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                
                                # Thêm padding
                                padding = 20
                                h, w = frame.shape[:2]
                                x1 = max(0, x1 - padding)
                                y1 = max(0, y1 - padding)
                                x2 = min(w, x2 + padding)
                                y2 = min(h, y2 + padding)
                                
                                # Cắt khuôn mặt
                                face = frame[y1:y2, x1:x2]
                                
                                if face.size > 0:
                                    face_frames.append(face)
            
            cap.release()
            
            if len(face_frames) == 0:
                print(f"⚠️ Không tìm thấy khuôn mặt trong video")
            
            return face_frames
            
        except Exception as e:
            print(f"❌ Lỗi khi xử lý video: {e}")
            return []
        