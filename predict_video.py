import tensorflow as tf
import cv2
import numpy as np
import os
from ai_model.config import *

MODEL_PATH = os.path.join(OUTPUT_DIR, MODEL_FILE)

def predict_video(video_path):
    """Chạy mô hình AI xem video là FAKE hay REAL (hiển thị cửa sổ)."""
    if not os.path.exists(MODEL_PATH):
        print(f"Lỗi: Không tìm thấy mô hình tại {MODEL_PATH}")
        return

    model = tf.keras.models.load_model(MODEL_PATH)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Lỗi: Không thể mở video.")
        return

    preds = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        try:
            face_crop = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
            face_array = np.asarray(face_crop, dtype=np.float32) / 255.0
            face_array = np.expand_dims(face_array, axis=0)
            prediction = model.predict(face_array, verbose=0)[0]
            fake_prob = float(prediction[1])
            preds.append(fake_prob)
        except Exception:
            continue

        cv2.imshow("Deepfake Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Phân tích hoàn tất!")

def predict_video_flask(video_path):
    """Hàm được Flask gọi, không hiển thị cửa sổ."""
    if not os.path.exists(MODEL_PATH):
        return {"label": "ERROR", "confidence": 0.0, "msg": f"Model not found at {MODEL_PATH}"}

    model = tf.keras.models.load_model(MODEL_PATH)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"label": "ERROR", "confidence": 0.0, "msg": "Cannot open video"}

    preds = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        try:
            face_crop = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
            face_array = np.asarray(face_crop, dtype=np.float32) / 255.0
            face_array = np.expand_dims(face_array, axis=0)
            prediction = model.predict(face_array, verbose=0)[0]
            fake_prob = float(prediction[1])
            preds.append(fake_prob)
        except Exception:
            continue

    cap.release()
    if len(preds) == 0:
        return {"label": "ERROR", "confidence": 0.0, "msg": "No frames processed"}

    avg_fake = np.mean(preds)
    label = "FAKE" if avg_fake > 0.5 else "REAL"
    return {"label": label, "confidence": round(float(avg_fake), 4)}
