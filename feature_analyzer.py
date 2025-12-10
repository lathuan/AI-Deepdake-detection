# utils/feature_analyzer.py
import cv2
import numpy as np
from scipy import ndimage

class DeepfakeFeatureAnalyzer:
    """
    Phân tích các đặc điểm để phát hiện deepfake:
    - Độ mịn bất thường của da
    - Sự không nhất quán ánh sáng
    - Ranh giới khuôn mặt mờ
    - Chi tiết mắt/răng
    - Artifacts nén
    """
    
    def __init__(self):
        self.suspicious_features = []
        
    def analyze_frame(self, face_image, confidence, heatmap=None):
        """
        Phân tích một frame và trả về danh sách các dấu hiệu nghi ngờ
        
        Args:
            face_image: numpy array của khuôn mặt (BGR)
            confidence: confidence score từ model (0-1)
            heatmap: attention map từ Grad-CAM
            
        Returns:
            dict: {
                'features': [list of feature dicts],
                'summary': str,
                'risk_level': 'high'|'medium'|'low'
            }
        """
        features = []
        
        # 1. Phân tích độ mịn da (skin smoothness)
        skin_analysis = self._analyze_skin_texture(face_image)
        if skin_analysis['is_suspicious']:
            features.append({
                'name': 'Độ mịn da bất thường',
                'description': 'Da quá mịn, thiếu chi tiết tự nhiên như lỗ chân lông, nếp nhăn nhỏ',
                'severity': skin_analysis['severity'],
                'confidence': skin_analysis['score']
            })
        
        # 2. Phân tích ranh giới khuôn mặt
        edge_analysis = self._analyze_face_boundaries(face_image)
        if edge_analysis['is_suspicious']:
            features.append({
                'name': 'Ranh giới khuôn mặt mờ',
                'description': 'Viền khuôn mặt có sự chuyển tiếp không tự nhiên, có thể do ghép ảnh',
                'severity': edge_analysis['severity'],
                'confidence': edge_analysis['score']
            })
        
        # 3. Phân tích vùng mắt
        eye_analysis = self._analyze_eye_region(face_image, heatmap)
        if eye_analysis['is_suspicious']:
            features.append({
                'name': 'Vùng mắt bất thường',
                'description': 'Mắt thiếu chi tiết, ánh mắt không tự nhiên, hoặc nhấp nháy không đồng bộ',
                'severity': eye_analysis['severity'],
                'confidence': eye_analysis['score']
            })
        
        # 4. Phân tích ánh sáng
        lighting_analysis = self._analyze_lighting_consistency(face_image)
        if lighting_analysis['is_suspicious']:
            features.append({
                'name': 'Ánh sáng không nhất quán',
                'description': 'Hướng và cường độ ánh sáng không đồng nhất trên khuôn mặt',
                'severity': lighting_analysis['severity'],
                'confidence': lighting_analysis['score']
            })
        
        # 5. Phân tích artifacts nén
        compression_analysis = self._analyze_compression_artifacts(face_image)
        if compression_analysis['is_suspicious']:
            features.append({
                'name': 'Artifacts nén bất thường',
                'description': 'Có dấu hiệu nén kỹ thuật số không đồng đều, có thể do xử lý AI',
                'severity': compression_analysis['severity'],
                'confidence': compression_analysis['score']
            })
        
        # 6. Phân tích từ heatmap
        if heatmap is not None:
            heatmap_analysis = self._analyze_heatmap_patterns(heatmap)
            if heatmap_analysis['is_suspicious']:
                features.append({
                    'name': 'Vùng tập trung bất thường',
                    'description': f'Model tập trung vào: {heatmap_analysis["focus_areas"]}',
                    'severity': heatmap_analysis['severity'],
                    'confidence': heatmap_analysis['score']
                })
        
        # Xác định risk level
        if confidence > 0.7:
            risk_level = 'high'
        elif confidence > 0.5:
            risk_level = 'medium'
        else:
            risk_level = 'low'
        
        # Tạo summary
        summary = self._create_summary(features, confidence, risk_level)
        
        return {
            'features': features,
            'summary': summary,
            'risk_level': risk_level,
            'num_suspicious_features': len(features)
        }
    
    def _analyze_skin_texture(self, face_image):
        """Phân tích texture da"""
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        
        # Tính variance của texture (da thật có variance cao hơn)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture_variance = laplacian.var()
        
        # Phát hiện da quá mịn (variance thấp bất thường)
        is_suspicious = texture_variance < 50  # Threshold có thể điều chỉnh
        
        severity = 'high' if texture_variance < 30 else 'medium' if texture_variance < 50 else 'low'
        score = max(0, min(1, (50 - texture_variance) / 50))
        
        return {
            'is_suspicious': is_suspicious,
            'severity': severity,
            'score': score,
            'texture_variance': texture_variance
        }
    
    def _analyze_face_boundaries(self, face_image):
        """Phân tích ranh giới khuôn mặt"""
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        
        # Detect edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Tính độ mạnh của edges ở viền
        h, w = edges.shape
        border_width = 20
        border_edges = np.concatenate([
            edges[:border_width, :].flatten(),
            edges[-border_width:, :].flatten(),
            edges[:, :border_width].flatten(),
            edges[:, -border_width:].flatten()
        ])
        
        border_edge_density = np.mean(border_edges > 0)
        
        # Ranh giới mờ = ít edges ở viền
        is_suspicious = border_edge_density < 0.05
        
        severity = 'high' if border_edge_density < 0.03 else 'medium'
        score = max(0, min(1, (0.05 - border_edge_density) / 0.05))
        
        return {
            'is_suspicious': is_suspicious,
            'severity': severity,
            'score': score,
            'edge_density': border_edge_density
        }
    
    def _analyze_eye_region(self, face_image, heatmap):
        """Phân tích vùng mắt"""
        h, w = face_image.shape[:2]
        
        # Giả định vùng mắt ở 1/3 trên của khuôn mặt
        eye_region = face_image[int(h*0.2):int(h*0.5), :]
        gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
        
        # Tính độ sắc nét của vùng mắt
        eye_sharpness = cv2.Laplacian(gray_eye, cv2.CV_64F).var()
        
        # Kiểm tra nếu heatmap tập trung vào mắt
        heatmap_focus_on_eyes = False
        if heatmap is not None:
            eye_heatmap = heatmap[int(h*0.2):int(h*0.5), :]
            heatmap_focus_on_eyes = np.mean(eye_heatmap) > 0.6
        
        is_suspicious = eye_sharpness < 100 or heatmap_focus_on_eyes
        
        severity = 'high' if eye_sharpness < 50 else 'medium'
        score = max(0, min(1, (100 - eye_sharpness) / 100))
        
        return {
            'is_suspicious': is_suspicious,
            'severity': severity,
            'score': score,
            'eye_sharpness': eye_sharpness
        }
    
    def _analyze_lighting_consistency(self, face_image):
        """Phân tích tính nhất quán của ánh sáng"""
        # Chuyển sang LAB color space để phân tích luminance
        lab = cv2.cvtColor(face_image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        
        # Chia khuôn mặt thành 4 quadrant
        h, w = l_channel.shape
        quadrants = [
            l_channel[:h//2, :w//2],      # Top-left
            l_channel[:h//2, w//2:],      # Top-right
            l_channel[h//2:, :w//2],      # Bottom-left
            l_channel[h//2:, w//2:]       # Bottom-right
        ]
        
        # Tính mean brightness của mỗi quadrant
        quadrant_means = [np.mean(q) for q in quadrants]
        
        # Tính variance giữa các quadrant
        lighting_variance = np.var(quadrant_means)
        
        # Variance cao = ánh sáng không đồng đều
        is_suspicious = lighting_variance > 200
        
        severity = 'high' if lighting_variance > 400 else 'medium'
        score = min(1, lighting_variance / 400)
        
        return {
            'is_suspicious': is_suspicious,
            'severity': severity,
            'score': score,
            'lighting_variance': lighting_variance
        }
    
    def _analyze_compression_artifacts(self, face_image):
        """Phân tích artifacts nén"""
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        
        # Tính DCT (Discrete Cosine Transform) để phát hiện blocking artifacts
        dct = cv2.dct(np.float32(gray))
        
        # Phân tích high-frequency components
        high_freq = dct[gray.shape[0]//2:, gray.shape[1]//2:]
        high_freq_energy = np.sum(np.abs(high_freq))
        
        # Artifacts nén = energy cao ở high frequencies
        is_suspicious = high_freq_energy > 1000000
        
        severity = 'medium' if high_freq_energy > 1500000 else 'low'
        score = min(1, high_freq_energy / 2000000)
        
        return {
            'is_suspicious': is_suspicious,
            'severity': severity,
            'score': score,
            'high_freq_energy': high_freq_energy
        }
    
    def _analyze_heatmap_patterns(self, heatmap):
        """Phân tích patterns từ attention heatmap"""
        # Tìm vùng có giá trị cao nhất
        threshold = 0.7
        hot_spots = heatmap > threshold
        
        # Label connected components
        labeled, num_features = ndimage.label(hot_spots)
        
        # Xác định vị trí các vùng nóng
        focus_areas = []
        h, w = heatmap.shape
        
        if np.mean(heatmap[:h//3, :]) > 0.5:
            focus_areas.append("vùng trán và mắt")
        if np.mean(heatmap[h//3:2*h//3, w//3:2*w//3]) > 0.5:
            focus_areas.append("trung tâm khuôn mặt")
        if np.mean(heatmap[2*h//3:, :]) > 0.5:
            focus_areas.append("vùng miệng và cằm")
        
        # Nghi ngờ nếu có nhiều hơn 2 vùng tập trung
        is_suspicious = len(focus_areas) >= 2
        
        severity = 'high' if len(focus_areas) >= 3 else 'medium'
        score = min(1, len(focus_areas) / 3)
        
        return {
            'is_suspicious': is_suspicious,
            'severity': severity,
            'score': score,
            'focus_areas': ", ".join(focus_areas) if focus_areas else "không rõ"
        }
    
    def _create_summary(self, features, confidence, risk_level):
        """Tạo summary text"""
        if risk_level == 'high':
            base = f"⚠️ CẢNH BÁO: Video có khả năng FAKE cao ({confidence*100:.1f}%)"
        elif risk_level == 'medium':
            base = f"⚡ CHÚ Ý: Video có dấu hiệu đáng ngờ ({confidence*100:.1f}%)"
        else:
            base = f"✅ Video có vẻ REAL ({(1-confidence)*100:.1f}%)"
        
        if len(features) > 0:
            feature_names = [f['name'] for f in features[:3]]  # Top 3
            summary = base + f"\n\nPhát hiện {len(features)} dấu hiệu nghi ngờ: " + ", ".join(feature_names)
        else:
            summary = base + "\n\nKhông phát hiện dấu hiệu bất thường rõ ràng."
        
        return summary