# utils/video_processor.py (cập nhật)
import os
# utils/video_processor.py
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from .face_detector import FaceDetector

class VideoProcessor:
    def __init__(self, face_detector, frame_size=224):
        self.face_detector = face_detector
        self.frame_size = frame_size
    
    def preprocess_faces(self, faces, max_frames=20):
        """Tiền xử lý các khuôn mặt"""
        if len(faces) == 0:
            faces = [np.zeros((self.frame_size, self.frame_size, 3), dtype=np.uint8)]
        
        if len(faces) > max_frames:
            faces = faces[:max_frames]
        elif len(faces) < max_frames:
            faces = faces * (max_frames // len(faces) + 1)
            faces = faces[:max_frames]
        
        face_tensors = []
        for face in faces:
            face_resized = cv2.resize(face, (self.frame_size, self.frame_size))
            face_normalized = face_resized.astype(np.float32) / 255.0
            face_tensor = torch.from_numpy(face_normalized).permute(2, 0, 1)
            face_tensors.append(face_tensor)
        
        return torch.stack(face_tensors)
    
    def analyze_frames_individually(self, faces, model, device):
        """Phân tích từng frame riêng lẻ để lấy confidence"""
        frame_confidences = []
        
        for face in faces:
            # Preprocess từng frame
            face_resized = cv2.resize(face, (self.frame_size, self.frame_size))
            face_normalized = face_resized.astype(np.float32) / 255.0
            face_tensor = torch.from_numpy(face_normalized).permute(2, 0, 1)
            face_tensor = face_tensor.unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 3, 224, 224)
            
            # Dự đoán cho từng frame
            with torch.no_grad():
                output = model(face_tensor)
                probability = torch.sigmoid(output).item()
                frame_confidences.append(probability)
        
        return frame_confidences
    
    def create_heatmap_overlay(self, face, heatmap, alpha=0.5):
        """Tạo heatmap overlay lên khuôn mặt"""
        # Chuẩn hóa heatmap
        heatmap = (heatmap * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Resize heatmap về kích thước face
        heatmap_resized = cv2.resize(heatmap_colored, (face.shape[1], face.shape[0]))
        
        # Blend với ảnh gốc
        overlay = cv2.addWeighted(face, 1-alpha, heatmap_resized, alpha, 0)
        return overlay
    
    def get_most_suspicious_frames(self, faces, frame_confidences, top_k=5):
        """Lấy các frame có confidence FAKE cao nhất"""
        frame_scores = list(zip(range(len(faces)), faces, frame_confidences))
        frame_scores.sort(key=lambda x: x[2], reverse=True)  # Sắp xếp theo confidence giảm dần
        return frame_scores[:top_k]
    
    def predict_video_detailed(self, video_path, model, device, max_frames=20):
        """
        Dự đoán chi tiết với phân tích từng frame
        """
        print(f"🎬 Đang phân tích video: {video_path}")
        
        # Trích xuất khuôn mặt
        faces = self.face_detector.extract_faces_from_video(video_path, max_frames)
        
        if len(faces) == 0:
            return {
                'prediction': 'UNKNOWN',
                'confidence': 0.5,
                'probability': 0.5,
                'num_faces': 0,
                'error': 'Không tìm thấy khuôn mặt',
                'frame_analysis': [],
                'time_confidence_data': []
            }
        
        # Phân tích toàn bộ video
        faces_tensor = self.preprocess_faces(faces, max_frames)
        faces_tensor_batch = faces_tensor.unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(faces_tensor_batch)
            overall_probability = torch.sigmoid(outputs).item()
        
        # Phân tích từng frame riêng lẻ
        frame_confidences = self.analyze_frames_individually(faces, model, device)
        
        # Lấy các frame nghi ngờ nhất
        suspicious_frames = self.get_most_suspicious_frames(faces, frame_confidences, top_k=5)
        
        # Tạo dữ liệu cho biểu đồ thời gian
        time_confidence_data = [
            {'frame_idx': i, 'time_sec': i * 0.2, 'confidence': conf}  # Giả sử 5fps
            for i, conf in enumerate(frame_confidences)
        ]
        
        # Lấy attention maps (nếu model hỗ trợ)
        attention_maps = []
        if hasattr(model, 'get_attention_maps'):
            try:
                attention_maps = model.get_attention_maps(faces_tensor_batch)
            except:
                attention_maps = [np.zeros((224, 224)) for _ in range(len(faces))]
        
        # Chuẩn bị kết quả chi tiết cho từng frame
        frame_analysis = []
        for idx, (frame_idx, face, confidence) in enumerate(suspicious_frames):
            frame_info = {
                'frame_index': frame_idx,
                'confidence': confidence,
                'face_image': face,
                'is_suspicious': confidence > 0.5
            }
            
            # Thêm heatmap nếu có
            if idx < len(attention_maps):
                heatmap = attention_maps[frame_idx] if frame_idx < len(attention_maps) else attention_maps[0]
                frame_info['heatmap_overlay'] = self.create_heatmap_overlay(face, heatmap)
                frame_info['heatmap'] = heatmap
            
            frame_analysis.append(frame_info)
        
        # Xác định kết quả tổng
        prediction = "FAKE" if overall_probability > 0.5 else "REAL"
        overall_confidence = overall_probability if prediction == "FAKE" else 1 - overall_probability
        
        return {
            'prediction': prediction,
            'confidence': overall_confidence,
            'probability': overall_probability,
            'num_faces': len(faces),
            'frame_analysis': frame_analysis,
            'time_confidence_data': time_confidence_data,
            'all_frame_confidences': frame_confidences
        }