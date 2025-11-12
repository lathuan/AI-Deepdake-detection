# predict.py
import os
import psycopg2
from utils.model_loader import load_trained_model
from utils.video_processor import process_video_for_api

# Load model gốc
MODEL_PATH = "model/best_deepfake_model_dfd.pth"
model, device = load_trained_model(MODEL_PATH)

# PostgreSQL connection (Render DATABASE_URL)
DB_URL = os.environ.get("DATABASE_URL")
conn = psycopg2.connect(DB_URL, sslmode="require")

def predict_deepfake(video_path):
    result = process_video_for_api(video_path, model, device)

    # Lưu kết quả vào DB
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO results (filename, prediction, probability) VALUES (%s, %s, %s)",
                (os.path.basename(video_path), result['prediction'], result['probability'])
            )
            conn.commit()
    except Exception as e:
        print(f"❌ Lỗi lưu DB: {e}")

    return result
