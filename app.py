# app.py
from flask import Flask, render_template, request, jsonify
import os
from predict import predict_deepfake
import psycopg2
from psycopg2.extras import RealDictCursor

app = Flask(__name__)
UPLOAD_FOLDER = "examples/test_videos"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Kết nối PostgreSQL
DB_URL = os.environ.get("DATABASE_URL")
conn = psycopg2.connect(DB_URL, sslmode="require")

# Khởi tạo bảng nếu chưa có
def init_db():
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS results (
                id SERIAL PRIMARY KEY,
                filename TEXT,
                prediction TEXT,
                probability FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.commit()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    video_file = request.files["video"]
    if video_file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    save_path = os.path.join(app.config["UPLOAD_FOLDER"], video_file.filename)
    video_file.save(save_path)

    result = predict_deepfake(save_path)
    os.remove(save_path)

    return jsonify(result)

@app.route("/results")
def get_results():
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("SELECT * FROM results ORDER BY created_at DESC LIMIT 20;")
        rows = cur.fetchall()
    return jsonify(rows)

if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
