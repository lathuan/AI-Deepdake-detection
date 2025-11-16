# utils/model_loader.py
import torch
import torch.nn as nn
import torchvision.models as models
import os
import cv2
import numpy as np

class ImprovedDeepfakeClassifier(nn.Module):
    def __init__(self, num_frames=20, num_classes=1):
        super(ImprovedDeepfakeClassifier, self).__init__()
        
        resnet = models.resnet18(pretrained=False)
        self.cnn_backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Lưu feature maps cho Grad-CAM
        self.feature_maps = None
        self.gradients = None
        
        # Hook để lấy feature maps
        def save_feature_maps(module, input, output):
            self.feature_maps = output
            
        self.cnn_backbone[-2].register_forward_hook(save_feature_maps)  # Layer trước avgpool
        
        def save_gradients(module, grad_in, grad_out):
            self.gradients = grad_out[0]
            
        self.cnn_backbone[-2].register_backward_hook(save_gradients)
        
        self.lstm = nn.LSTM(512, 128, batch_first=True, bidirectional=True, dropout=0.3)
        self.classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        batch_size, num_frames, C, H, W = x.shape
        cnn_features = []
        
        for i in range(num_frames):
            frame = x[:, i, :, :, :]
            features = self.cnn_backbone(frame)
            features = features.view(batch_size, -1)
            cnn_features.append(features)
        
        cnn_features = torch.stack(cnn_features, dim=1)
        lstm_out, _ = self.lstm(cnn_features)
        output = self.classifier(lstm_out[:, -1, :])
        return output
    
    def get_attention_maps(self, x):
        """Lấy attention maps cho từng frame sử dụng Grad-CAM"""
        batch_size, num_frames, C, H, W = x.shape
        attention_maps = []
        
        for i in range(num_frames):
            frame = x[:, i:i+1, :, :, :]  # Giữ batch dimension
            
            # Forward pass
            self.zero_grad()
            output = self.forward(frame)
            
            # Backward pass để lấy gradients
            output.backward(torch.ones_like(output))
            
            # Tính Grad-CAM
            if self.gradients is not None and self.feature_maps is not None:
                pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
                
                # Weighted combination of feature maps
                feature_maps = self.feature_maps.squeeze(0)
                cam = torch.zeros(feature_maps.shape[1:], dtype=torch.float32)
                
                for j, weight in enumerate(pooled_gradients):
                    cam += weight * feature_maps[j, :, :]
                
                # ReLU và resize
                cam = torch.relu(cam)
                cam = cam.detach().cpu().numpy()
                cam = cv2.resize(cam, (224, 224))
                cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
                
                attention_maps.append(cam)
            else:
                attention_maps.append(np.zeros((224, 224)))
        
        return attention_maps

def load_trained_model(model_path, device='auto'):
    """Load model đã train từ file .pth"""
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Không tìm thấy file model: {model_path}")
    
    model = ImprovedDeepfakeClassifier()
    checkpoint = torch.load(model_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print(f"✅ Đã load model từ: {model_path}")
    print(f"🖥️ Device: {device}")
    
    return model, device