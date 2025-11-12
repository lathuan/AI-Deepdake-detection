# utils/video_processor.py
import cv2
import numpy as np
import torch
from .face_detector import FaceDetector

class VideoProcessor:
    def __init__(self, face_detector, frame_size=224):
        self.face_detector = face_detector
        self.frame_size = frame_size

    def preprocess_faces(self, faces, max_frames=20):
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

    def predict_video(self, video_path, model, device, max_frames=20):
        faces = self.face_detector.extract_faces_from_video(video_path, max_frames)
        if len(faces) == 0:
            return {
                'prediction': 'UNKNOWN',
                'confidence': 0.5,
                'probability': 0.5,
                'num_faces': 0,
                'error': 'Không tìm thấy khuôn mặt'
            }

        faces_tensor = self.preprocess_faces(faces, max_frames)
        faces_tensor = faces_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(faces_tensor)
            probability = torch.sigmoid(outputs).item()

        prediction = "FAKE" if probability > 0.5 else "REAL"
        confidence = probability if prediction == "FAKE" else 1 - probability

        return {
            'prediction': prediction,
            'confidence': confidence,
            'probability': probability,
            'num_faces': len(faces),
            'faces_sample': faces[:3]
        }

# Hàm tiện ích cho Flask API
def process_video_for_api(video_path, model, device):
    detector = FaceDetector()
    processor = VideoProcessor(detector)
    return processor.predict_video(video_path, model, device)
