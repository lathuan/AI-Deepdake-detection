# utils/face_detector.py (cập nhật thông báo tiến trình)
from ultralytics import YOLO
import cv2
import os

class FaceDetector:
    def __init__(self, model_path='yolov8l-face-lindevs.pt'):  # Dùng model nhẹ hơn cho local
        if not os.path.exists(model_path):
            print(f"📥 Đang tải model face detection...")
            try:
                self.model = YOLO('yolov8l-face-lindevs.pt')
            except:
                # Fallback to smaller model
                self.model = YOLO('yolov8n.pt')
        else:
            self.model = YOLO(model_path)
        
        print(f"✅ Face detector ready!")

    def extract_faces_from_video(self, video_path, max_frames=20, conf_threshold=0.7):
        """
        Trích xuất khuôn mặt từ video với progress reporting
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"❌ Không thể mở video: {video_path}")
                return []
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"📊 Video có {total_frames} frames tổng cộng")
            
            face_frames = []
            frame_count = 0
            processed_count = 0
            max_frames_to_process = min(100, total_frames)
            
            while len(face_frames) < max_frames and frame_count < max_frames_to_process:
                ret, frame = cap.read()
                if not ret:
                    frame_count += 1
                
                # Hiển thị tiến độ mỗi 20 frames
                if frame_count % 20 == 0:
                    print(f"⏳ Đã xử lý {frame_count}/{max_frames_to_process} frames...")
                
                # Xử lý mỗi 5 frame
                if frame_count % 5 != 0:
                    continue
                
                processed_count += 1
                
                # Phát hiện khuôn mặt
                results = self.model(frame, conf=conf_threshold, verbose=False)
                
                for result in results:
                    if len(result.boxes) > 0:
                        for box in result.boxes:
                            confidence = box.conf[0].item()
                            if confidence > conf_threshold:
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                padding = 20
                                h, w = frame.shape[:2]
                                x1 = max(0, x1 - padding)
                                y1 = max(0, y1 - padding)
                                x2 = min(w, x2 + padding)
                                y2 = min(h, y2 + padding)
                                
                                face = frame[y1:y2, x1:x2]
                                if face.size > 0:
                                    face_frames.append(face)
            
            cap.release()
            
            print(f"✅ Đã tìm thấy {len(face_frames)} khuôn mặt từ {processed_count} frames đã xử lý")
            return face_frames
            
        except Exception as e:
            print(f"❌ Lỗi khi xử lý video: {e}")
            return []