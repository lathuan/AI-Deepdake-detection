import os
import uuid
from pathlib import Path
from flask import Flask, render_template, request, url_for, send_from_directory, session, jsonify
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
import cv2
import numpy as np

# Import hàm AI
from ai_model.predict_video import predict_video_flask

BASE_DIR = Path(__file__).parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)

app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = os.environ.get("FLASK_SECRET", "dev-secret-change-me")

# DB config: DATABASE_URL (Postgres) nếu có, ngược lại dùng SQLite
database_url = os.environ.get("DATABASE_URL") or f"sqlite:///{BASE_DIR / 'app.db'}"
app.config["SQLALCHEMY_DATABASE_URI"] = database_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Giới hạn kích thước upload (500MB)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)

ALLOWED_EXT = {".mp4", ".mov", ".webm", ".ogg", ".mkv"}

db = SQLAlchemy(app)

# ---------- Models ----------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(300), nullable=False)

# ---------- Utilities ----------
def allowed_file(filename):
    ext = Path(filename).suffix.lower()
    return ext in ALLOWED_EXT

def save_upload(file_storage):
    filename = secure_filename(file_storage.filename)
    unique = f"{uuid.uuid4().hex}_{filename}"
    outpath = UPLOAD_FOLDER / unique
    file_storage.save(str(outpath))
    return unique, outpath

# ---------- Routes ----------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

@app.route("/analyze", methods=["POST"])
def analyze():
    if "video" not in request.files:
        return jsonify({"error": "No file part"}), 400
    f = request.files["video"]
    if f.filename == "":
        return jsonify({"error": "No selected file"}), 400
    if not allowed_file(f.filename):
        return jsonify({"error": "File type not allowed"}), 400

    unique_name, out_path = save_upload(f)
    try:
        result_ai = predict_video_flask(str(out_path))
        label = "Suspicious (deepfake suspected)" if result_ai["label"] == "FAKE" else "Likely real"
        result = {
            "label": label,
            "score": result_ai["confidence"],
            "video_url": url_for("uploaded_file", filename=unique_name)
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------- Auth ----------
@app.route("/register", methods=["POST"])
def register():
    data = request.get_json(force=True)
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""
    if not username or not password:
        return jsonify({"msg": "username and password required"}), 400
    if User.query.filter_by(username=username).first():
        return jsonify({"msg": "username exists"}), 400
    u = User(username=username, password_hash=generate_password_hash(password))
    db.session.add(u)
    db.session.commit()
    return jsonify({"msg": "registered"})

@app.route("/login", methods=["POST"])
def login():
    data = request.get_json(force=True)
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""
    u = User.query.filter_by(username=username).first()
    if not u or not check_password_hash(u.password_hash, password):
        return jsonify({"msg": "invalid credentials"}), 401
    session["user_id"] = u.id
    return jsonify({"msg": "ok"})

# ---------- Manual DB init route (cho Render Free plan) ----------
@app.route("/initdb")
def initdb_route():
    try:
        with app.app_context():
            db.create_all()
        return "✅ Database initialized successfully!"
    except Exception as e:
        return f"❌ Error initializing DB: {e}"

# ---------- CLI ----------
@app.cli.command("init-db")
def init_db():
    db.create_all()
    print("DB initialized.")

if __name__ == "__main__":
    if "sqlite" in database_url and not (BASE_DIR / "app.db").exists():
        with app.app_context():
            db.create_all()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
