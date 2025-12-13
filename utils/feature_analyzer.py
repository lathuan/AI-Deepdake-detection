# utils/feature_analyzer.py
import cv2
import numpy as np
from scipy import ndimage

class DeepfakeFeatureAnalyzer:
    """
    Analyze features to detect deepfakes:
    - Abnormal skin smoothness
    - Lighting inconsistency
    - Blurred face boundaries
    - Eye/teeth details
    - Compression artifacts
    """
    
    def __init__(self):
        self.suspicious_features = []
        
    def analyze_frame(self, face_image, confidence, heatmap=None):
        """
        Analyze a frame and return list of suspicious signs
        
        Args:
            face_image: numpy array of face (BGR)
            confidence: confidence score from model (0-1)
            heatmap: attention map from Grad-CAM
            
        Returns:
            dict: {
                'features': [list of feature dicts],
                'summary': str,
                'risk_level': 'high'|'medium'|'low'
            }
        """
        features = []
        
        # 1. Analyze skin smoothness
        skin_analysis = self._analyze_skin_texture(face_image)
        if skin_analysis['is_suspicious']:
            features.append({
                'name': 'Abnormal skin smoothness',
                'description': 'Skin is overly smooth, lacking natural details like pores or fine wrinkles',
                'severity': skin_analysis['severity'],
                'confidence': skin_analysis['score']
            })
        
        # 2. Analyze face boundaries
        edge_analysis = self._analyze_face_boundaries(face_image)
        if edge_analysis['is_suspicious']:
            features.append({
                'name': 'Blurred face boundaries',
                'description': 'Face edges show unnatural transitions, possibly due to compositing',
                'severity': edge_analysis['severity'],
                'confidence': edge_analysis['score']
            })
        
        # 3. Analyze eye region
        eye_analysis = self._analyze_eye_region(face_image, heatmap)
        if eye_analysis['is_suspicious']:
            features.append({
                'name': 'Abnormal eye region',
                'description': 'Eyes lack detail, unnatural gaze, or inconsistent blinking',
                'severity': eye_analysis['severity'],
                'confidence': eye_analysis['score']
            })
        
        # 4. Analyze lighting
        lighting_analysis = self._analyze_lighting_consistency(face_image)
        if lighting_analysis['is_suspicious']:
            features.append({
                'name': 'Inconsistent lighting',
                'description': 'Lighting direction and intensity are inconsistent across the face',
                'severity': lighting_analysis['severity'],
                'confidence': lighting_analysis['score']
            })
        
        # 5. Analyze compression artifacts
        compression_analysis = self._analyze_compression_artifacts(face_image)
        if compression_analysis['is_suspicious']:
            features.append({
                'name': 'Unusual compression artifacts',
                'description': 'Signs of uneven digital compression, possibly from AI processing',
                'severity': compression_analysis['severity'],
                'confidence': compression_analysis['score']
            })
        
        # 6. Analyze from heatmap
        if heatmap is not None:
            heatmap_analysis = self._analyze_heatmap_patterns(heatmap)
            if heatmap_analysis['is_suspicious']:
                features.append({
                    'name': 'Unusual focus areas',
                    'description': f'Model focused on: {heatmap_analysis["focus_areas"]}',
                    'severity': heatmap_analysis['severity'],
                    'confidence': heatmap_analysis['score']
                })
        
        # Determine risk level
        if confidence > 0.7:
            risk_level = 'high'
        elif confidence > 0.5:
            risk_level = 'medium'
        else:
            risk_level = 'low'
        
        # Create summary (English)
        summary = self._create_summary(features, confidence, risk_level)
        
        return {
            'features': features,
            'summary': summary,
            'risk_level': risk_level,
            'num_suspicious_features': len(features)
        }
    
    def _analyze_skin_texture(self, face_image):
        """Analyze skin texture"""
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        
        # Calculate texture variance (real skin has higher variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture_variance = laplacian.var()
        
        # Detect overly smooth skin (low variance)
        is_suspicious = texture_variance < 50  # Threshold can be adjusted
        
        severity = 'high' if texture_variance < 30 else 'medium' if texture_variance < 50 else 'low'
        score = max(0, min(1, (50 - texture_variance) / 50))
        
        return {
            'is_suspicious': is_suspicious,
            'severity': severity,
            'score': score,
            'texture_variance': texture_variance
        }
    
    def _analyze_face_boundaries(self, face_image):
        """Analyze face boundaries"""
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        
        # Detect edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Calculate edge strength at borders
        h, w = edges.shape
        border_width = 20
        border_edges = np.concatenate([
            edges[:border_width, :].flatten(),
            edges[-border_width:, :].flatten(),
            edges[:, :border_width].flatten(),
            edges[:, -border_width:].flatten()
        ])
        
        border_edge_density = np.mean(border_edges > 0)
        
        # Blurred boundaries = low edges at borders
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
        """Analyze eye region"""
        h, w = face_image.shape[:2]
        
        # Assume eye region in top 1/3 of face
        eye_region = face_image[int(h*0.2):int(h*0.5), :]
        gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
        
        # Calculate sharpness of eye region
        eye_sharpness = cv2.Laplacian(gray_eye, cv2.CV_64F).var()
        
        # Check if heatmap focuses on eyes
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
        """Analyze lighting consistency"""
        # Convert to LAB for luminance
        lab = cv2.cvtColor(face_image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        
        # Divide face into 4 quadrants
        h, w = l_channel.shape
        quadrants = [
            l_channel[:h//2, :w//2],      # Top-left
            l_channel[:h//2, w//2:],      # Top-right
            l_channel[h//2:, :w//2],      # Bottom-left
            l_channel[h//2:, w//2:]       # Bottom-right
        ]
        
        # Calculate mean brightness per quadrant
        quadrant_means = [np.mean(q) for q in quadrants]
        
        # Calculate variance between quadrants
        lighting_variance = np.var(quadrant_means)
        
        # High variance = inconsistent lighting
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
        """Analyze compression artifacts"""
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        
        # DCT for blocking artifacts
        dct = cv2.dct(np.float32(gray))
        
        # High-frequency components
        high_freq = dct[gray.shape[0]//2:, gray.shape[1]//2:]
        high_freq_energy = np.sum(np.abs(high_freq))
        
        # Compression artifacts = high energy in high freq
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
        """Analyze patterns from attention heatmap"""
        # Find hot spots
        threshold = 0.7
        hot_spots = heatmap > threshold
        
        # Label connected components
        labeled, num_features = ndimage.label(hot_spots)
        
        # Identify focus areas
        focus_areas = []
        h, w = heatmap.shape
        
        if np.mean(heatmap[:h//3, :]) > 0.5:
            focus_areas.append("forehead and eyes")
        if np.mean(heatmap[h//3:2*h//3, w//3:2*w//3]) > 0.5:
            focus_areas.append("face center")
        if np.mean(heatmap[2*h//3:, :]) > 0.5:
            focus_areas.append("mouth and chin")
        
        # Suspicious if more than 2 focus areas
        is_suspicious = len(focus_areas) >= 2
        
        severity = 'high' if len(focus_areas) >= 3 else 'medium'
        score = min(1, len(focus_areas) / 3)
        
        return {
            'is_suspicious': is_suspicious,
            'severity': severity,
            'score': score,
            'focus_areas': ", ".join(focus_areas) if focus_areas else "unclear"
        }
    
    def _create_summary(self, features, confidence, risk_level):
        """Create summary text (English)"""
        if risk_level == 'high':
            base = f"⚠️ WARNING: Video has high FAKE probability ({confidence*100:.1f}%)"
        elif risk_level == 'medium':
            base = f"⚡ ATTENTION: Video shows suspicious signs ({confidence*100:.1f}%)"
        else:
            base = f"✅ Video appears REAL ({(1-confidence)*100:.1f}%)"
        
        if len(features) > 0:
            feature_names = [f['name'] for f in features[:3]]  # Top 3
            summary = base + f"\n\nDetected {len(features)} suspicious features: " + ", ".join(feature_names)
        else:
            summary = base + "\n\nNo clear anomalies detected."
        
        return summary