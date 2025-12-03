
# face_detector.py - Sử dụng MediaPipe thay vì Haar Cascade

import cv2
import numpy as np
import mediapipe as mp


class FaceDetector:
    """
    Lớp phát hiện khuôn mặt sử dụng MediaPipe (tốt hơn Haar Cascade)
    """
    
    def __init__(self, min_detection_confidence=0.5):
        """
        Khởi tạo detector
        
        Args:
            min_detection_confidence: Ngưỡng tin cậy tối thiểu (0.0 - 1.0)
        """
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,  # 0 = short-range (2m), 1 = full-range (5m)
            min_detection_confidence=min_detection_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
    
    def detect_faces(self, frame):
        """
        Phát hiện khuôn mặt trong frame
        
        Args:
            frame: Image RGB (H, W, 3)
        
        Returns:
            List các bounding box: [(x, y, w, h), ...]
        """
        results = self.face_detection.process(frame)
        
        faces = []
        if results.detections:
            h, w, _ = frame.shape
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                
                # Chuyển đổi từ tọa độ tương đối thành pixel
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                box_w = int(bbox.width * w)
                box_h = int(bbox.height * h)
                
                # Đảm bảo không vượt ra ngoài frame
                x = max(0, x)
                y = max(0, y)
                box_w = min(box_w, w - x)
                box_h = min(box_h, h - y)
                
                if box_w > 0 and box_h > 0:
                    faces.append((x, y, box_w, box_h))
        
        return faces
    
    def extract_face_region(self, frame, face_coords, padding=0.2):
        """
        Trích xuất vùng khuôn mặt với padding
        
        Args:
            frame: Input image
            face_coords: (x, y, w, h)
            padding: Tỷ lệ padding xung quanh khuôn mặt
        
        Returns:
            Cropped face region
        """
        x, y, w, h = face_coords
        pad_x = int(w * padding)
        pad_y = int(h * padding)
        
        x_start = max(0, x - pad_x)
        y_start = max(0, y - pad_y)
        x_end = min(frame.shape[1], x + w + pad_x)
        y_end = min(frame.shape[0], y + h + pad_y)
        
        face_region = frame[y_start:y_end, x_start:x_end]
        return face_region
    
    def detect_and_extract(self, frame, padding=0.2):
        """
        Phát hiện khuôn mặt và trích xuất tất cả các region
        
        Returns:
            List các face regions
        """
        faces = self.detect_faces(frame)
        
        if len(faces) == 0:
            # Nếu không phát hiện khuôn mặt, dùng toàn bộ frame
            return [frame]
        
        face_regions = []
        for face_coords in faces:
            face_region = self.extract_face_region(frame, face_coords, padding)
            if face_region.size > 0:
                face_regions.append(face_region)
        
        return face_regions if face_regions else [frame]
    
    def get_largest_face(self, frame, padding=0.2):
        """
        Trích xuất khuôn mặt lớn nhất (được sử dụng khi chỉ cần 1 khuôn mặt)
        
        Returns:
            Face region lớn nhất hoặc toàn bộ frame
        """
        faces = self.detect_faces(frame)
        
        if len(faces) == 0:
            return frame
        
        # Tìm khuôn mặt lớn nhất dựa trên diện tích
        largest_face = max(faces, key=lambda x: x[2] * x[3])
        face_region = self.extract_face_region(frame, largest_face, padding)
        
        return face_region if face_region.size > 0 else frame
    
    def draw_detections(self, frame, faces, color=(0, 255, 0), thickness=2):
        """
        Vẽ bounding box trên frame (dùng cho visualization)
        
        Args:
            frame: Input image
            faces: List bounding boxes
            color: Màu (B, G, R)
            thickness: Độ dày đường vẽ
        
        Returns:
            Frame với bounding boxes
        """
        frame_copy = frame.copy()
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame_copy, (x, y), (x + w, y + h), color, thickness)
        
        return frame_copy
    
    def release(self):
        """
        Giải phóng resources
        """
        if self.face_detection:
            self.face_detection.close()


# ===== EXAMPLE USAGE =====
if __name__ == '__main__':
    # Khởi tạo detector
    detector = FaceDetector(min_detection_confidence=0.5)
    
    # Test với webcam
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Phát hiện khuôn mặt
        faces = detector.detect_faces(frame_rgb)
        
        # Vẽ bounding boxes (cần convert frame về BGR lại)
        frame_with_boxes = detector.draw_detections(frame, faces)
        
        # Hiển thị
        cv2.imshow('Face Detection', frame_with_boxes)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    detector.release()