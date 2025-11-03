import os
import uuid
from pathlib import Path
from flask import (
    Flask, render_template, request, url_for, send_from_directory,
    session, jsonify, redirect, flash
)
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from models import db, User, VideoUpload, Prediction
from predict_video import predict_video_flask

BASE_DIR = Path(__file__).parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)

app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = os.environ.get("FLASK_SECRET", "dev-secret-change-me")

# DATABASE_URL injected by Render; fallback to local sqlite for dev
database_url = os.environ.get("DATABASE_URL") or f"sqlite:///{BASE_DIR/'app.db'}"
app.config["SQLALCHEMY_DATABASE_URI"] = database_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db.init_app(app)

# uploads
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
ALLOWED_EXT = {".mp4", ".mov", ".webm", ".ogg", ".mkv"}

def allowed_file(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXT

# ---------- Pages ----------
@app.route("/")
def index():
    return render_template("index.html", user=session.get("username"))

@app.route("/login")
def login_page():
    if session.get("user_id"):
        return redirect("/dashboard")
    return render_template("login.html")

@app.route("/register")
def register_page():
    if session.get("user_id"):
        return redirect("/dashboard")
    return render_template("register.html")

@app.route("/dashboard")
def dashboard():
    if not session.get("user_id"):
        return redirect("/login")
    user_id = session["user_id"]
    videos = VideoUpload.query.filter_by(user_id=user_id).order_by(VideoUpload.uploaded_at.desc()).all()
    return render_template("dashboard.html", videos=videos, user=session.get("username"))

# ---------- Static served uploads ----------
@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

# ---------- API: analyze ----------
@app.route("/analyze", methods=["POST"])
def analyze():
    if not session.get("user_id"):
        return jsonify({"error": "Not authenticated"}), 401

    if "video" not in request.files:
        return jsonify({"error": "No file part"}), 400
    f = request.files["video"]
    if f.filename == "":
        return jsonify({"error": "No selected file"}), 400
    if not allowed_file(f.filename):
        return jsonify({"error": "File type not allowed"}), 400

    # save file
    filename = secure_filename(f.filename)
    unique = f"{uuid.uuid4().hex}_{filename}"
    outpath = UPLOAD_FOLDER / unique
    f.save(str(outpath))

    # save record
    v = VideoUpload(filename=unique, filepath=str(outpath), user_id=session["user_id"])
    db.session.add(v)
    db.session.commit()

    # run AI (predict_video_flask returns dict {"label","confidence",...})
    result = predict_video_flask(str(outpath))

    # Save prediction
    pred = Prediction(video_id=v.id, label=result.get("label", "ERROR"), confidence=float(result.get("confidence", 0.0)))
    db.session.add(pred)
    db.session.commit()

    return jsonify({
        "label": result.get("label"),
        "score": result.get("confidence"),
        "video_url": url_for("uploaded_file", filename=unique)
    })

# ---------- Auth API ----------
@app.route("/api/register", methods=["POST"])
def api_register():
    data = request.get_json(force=True)
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""
    if not username or not password:
        return jsonify({"msg":"username and password required"}), 400
    if User.query.filter_by(username=username).first():
        return jsonify({"msg":"username exists"}), 400
    u = User(username=username, password=generate_password_hash(password))
    db.session.add(u)
    db.session.commit()
    return jsonify({"msg":"registered"})

@app.route("/api/login", methods=["POST"])
def api_login():
    data = request.get_json(force=True)
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""
    u = User.query.filter_by(username=username).first()
    if not u or not check_password_hash(u.password, password):
        return jsonify({"msg":"invalid credentials"}), 401
    session["user_id"] = u.id
    session["username"] = u.username
    return jsonify({"msg":"ok"})

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")

# ---------- DB init helpers ----------
@app.cli.command("init-db")
def init_db():
    with app.app_context():
        db.create_all()
    print("DB initialized.")

@app.route("/initdb")
def initdb_route():
    try:
        with app.app_context():
            db.create_all()
        return "✅ Database initialized successfully!"
    except Exception as e:
        return f"❌ Error initializing DB: {e}"

# ---------- Run ----------
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
