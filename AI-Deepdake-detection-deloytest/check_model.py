# check_model.py
import os
import torch

def check_model_file(model_path):
    print(f"🔍 Kiểm tra file model: {model_path}")
    
    if not os.path.exists(model_path):
        print("❌ File model không tồn tại!")
        return False
    
    file_size = os.path.getsize(model_path) / 1024 / 1024
    print(f"📏 Kích thước file: {file_size:.2f} MB")
    
    if file_size < 1:
        print("❌ File model quá nhỏ, có thể bị hỏng!")
        return False
    
    try:
        # Thử load với các phương pháp khác nhau
        print("🔄 Đang thử load model...")
        
        # Phương pháp 1: Load bình thường
        try:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            print("✅ Load thành công với phương pháp 1")
            return True
        except:
            pass
        
        # Phương pháp 2: Load với pickle (cẩn thận)
        try:
            checkpoint = torch.load(model_path, map_location='cpu', pickle_module=__import__('pickle'))
            print("✅ Load thành công với phương pháp 2")
            return True
        except:
            pass
            
        print("❌ Tất cả phương pháp load đều thất bại!")
        return False
        
    except Exception as e:
        print(f"❌ Lỗi khi kiểm tra model: {e}")
        return False

if __name__ == "__main__":
    check_model_file("model/best_deepfake_model_dfd.pth")