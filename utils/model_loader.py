# utils/model_loader.py
import torch
import torch.nn as nn
import torchvision.models as models
import os

class ImprovedDeepfakeClassifier(nn.Module):
    def __init__(self, num_frames=20, num_classes=1):
        super(ImprovedDeepfakeClassifier, self).__init__()
        
        resnet = models.resnet18(pretrained=False)
        self.cnn_backbone = nn.Sequential(*list(resnet.children())[:-1])
        
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

def load_trained_model(model_path, device='auto'):
    """
    Load model đã train từ file .pth
    
    Args:
        model_path: Đường dẫn đến file model
        device: 'auto', 'cuda', hoặc 'cpu'
    
    Returns:
        model: Model đã được load weights
    """
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Kiểm tra file model có tồn tại không
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Không tìm thấy file model: {model_path}")
    
    # Khởi tạo model
    model = ImprovedDeepfakeClassifier()
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    
    # Xử lý các định dạng checkpoint khác nhau
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Load state dict
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print(f"✅ Đã load model từ: {model_path}")
    print(f"🖥️ Device: {device}")
    
    return model, device