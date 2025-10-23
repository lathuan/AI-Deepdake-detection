# predict_video.py

import tensorflow as tf
import cv2
import numpy as np
import os
from config import * # Lưu ý: Cần thư viện khuôn mặt (ví dụ: OpenCV DNN, MediaPipe) để cắt khuôn mặt
# Phần này giả định bạn có logic trích xuất khuôn mặt (Face Detection)

MODEL_PATH = os.path.join(OUTPUT_DIR, MODEL_FILE)

def predict_video(video_path):
    if not os.path.exists(MODEL_PATH):
        print(f"Lỗi: Không tìm thấy mô hình tại {MODEL_PATH}. Vui lòng huấn luyện mô hình trước.")
        return

    # Tải mô hình đã huấn luyện
    model = tf.keras.models.load_model(MODEL_PATH)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Lỗi: Không thể mở video.")
        return

    # Khởi tạo Face Detector (GIẢ ĐỊNH)
    # Bạn sẽ cần tích hợp một face detector thực tế tại đây (ví dụ: Haarcascade, MTCNN, MediaPipe Face Detection)
    # Vì đây là ví dụ, chúng ta sẽ bỏ qua phần Face Detector phức tạp
    print("CHÚ Ý: Hàm này cần tích hợp Face Detector thực tế!")
    
    output_frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # --- BƯỚC 1: PHÁT HIỆN VÀ CẮT KHUÔN MẶT (GIẢ ĐỊNH) ---
        # Ví dụ: Giả định đã phát hiện được khuôn mặt tại (x, y, w, h)
        
        # CHỈ DỰ ĐOÁN VÀO KHUÔN MẶT ĐÃ CẮT (CROPPED FACE)
        # Trong môi trường thực tế, bạn cần cắt khuôn mặt tại đây
        
        # Giả định: frame đã được tiền xử lý thành face_array có kích thước (128, 128, 3)
        
        # Nếu không phát hiện khuôn mặt nào, thì bỏ qua frame
        
        # Để chạy thử nghiệm, chúng ta chỉ lấy một phần của frame
        # (Bạn phải thay thế bằng logic cắt khuôn mặt thực tế)
        try:
            face_crop = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
            face_array = np.asarray(face_crop, dtype=np.float32) / 255.0
            face_array = np.expand_dims(face_array, axis=0)
        except Exception:
             continue

        # --- BƯỚC 2: DỰ ĐOÁN ---
        prediction = model.predict(face_array)[0] 
        
        # 3. PHIÊN DỊCH KẾT QUẢ NHỊ PHÂN
        # prediction là [Xác suất REAL, Xác suất FAKE]
        real_prob = prediction[0]
        fake_prob = prediction[1]
        
        label = "REAL" if real_prob > fake_prob else "FAKE"
        prob = max(real_prob, fake_prob)
        
        # 4. Hiển thị kết quả trên Frame
        color = (0, 255, 0) if label == "REAL" else (0, 0, 255) # Xanh lá vs Đỏ
        text = f"{label}: {prob*100:.2f}%"
        
        # Vẽ văn bản lên frame (giả định vẽ ở góc trên)
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        
        output_frames.append(frame)
        cv2.imshow('Deepfake Detector', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    print(f"Phân tích video {video_path} hoàn tất.")


if __name__ == '__main__':
    # Bạn cần thay thế bằng đường dẫn video thực tế để kiểm tra
    # Ví dụ: python predict_video.py --video_path 'videos/test_fake.mp4'
    import sys
    if len(sys.argv) > 1:
        predict_video(sys.argv[1])
    else:
        print("Vui lòng cung cấp đường dẫn video: python predict_video.py duong/dan/video.mp4")