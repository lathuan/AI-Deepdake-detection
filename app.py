from flask import Flask, g, render_template, request, jsonify, session, redirect, url_for
import sqlite3
import os
import io
import base64

from werkzeug.security import generate_password_hash, check_password_hash
from PIL import Image
from predict import predict_deepfake

app = Flask(__name__)
app.secret_key = "YOUR_SECRET_KEY"

DATABASE = "deepfake_results.db"

# 🔥 FIX: Thêm hàm get_db() để mở kết nối SQLite
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DATABASE)
        g.db.row_factory = sqlite3.Row
    return g.db

# 🔥 FIX: Đóng database sau mỗi request
@app.teardown_appcontext
def close_db(exception):
    db = g.pop("db", None)
    if db is not None:
        db.close()

# Tạo bảng users
def init_db():
    db = get_db()
    db.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            email TEXT NOT NULL,
            password TEXT NOT NULL
        )
    """)
    db.commit()
if not os.path.exists(DATABASE):
    with app.app_context():
        init_db()



UPLOAD_FOLDER = "examples_test_videos"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/")
def index():
    return render_template("index.html")


# ------------------ AUTH ------------------

# REGISTER (GET + POST)
@app.route("/register", methods=["GET"])
def register_page():
    return render_template("register.html")


@app.route("/register", methods=["POST"])
def register_submit():
    name = request.form.get("name")
    email = request.form.get("email")
    password = request.form.get("password")

    if not name or not email or not password:
        return jsonify({"error": "Missing info"}), 400

    hashed_pw = generate_password_hash(password)

    db = get_db()
    try:
        db.execute(
            "INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
            (name, email, hashed_pw)
        )
        db.commit()
    except sqlite3.IntegrityError:
        return jsonify({"error": "Email already exists"}), 400

    return jsonify({"message": "Register success!"})



# LOGIN (GET + POST)
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        return render_template("login.html")

    email = request.form.get("email")
    password = request.form.get("password")

    db = get_db()
    user = db.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()

    if user is None:
        return "Sai tài khoản hoặc mật khẩu!"

    if not check_password_hash(user["password"], password):
        return "Sai tài khoản hoặc mật khẩu!"

    session["user_id"] = user["id"]
    session["email"] = user["email"]

    return redirect(url_for("index"))


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ------------------------------------------------


def pil_image_to_base64(img):
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def plot_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# ------------------ UPLOAD VIDEO ------------------
@app.route("/upload", methods=["POST"])
def upload_video():

    if not is_logged_in():
        return jsonify({"error": "Unauthorized"}), 403

    if "video" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    video_file = request.files["video"]
    if video_file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    save_path = os.path.join(app.config["UPLOAD_FOLDER"], video_file.filename)
    video_file.save(save_path)

    result = predict_deepfake(save_path)
    os.remove(save_path)

    if "error" in result:
        return jsonify(result), 400

    frames_base64 = []
    if "frames_for_web" in result:
        for frame in result["frames_for_web"]:
            face_b64 = pil_image_to_base64(frame["face_image"])
            heatmap_b64 = pil_image_to_base64(frame["heatmap_overlay"]) if frame["heatmap_overlay"] else None
            frames_base64.append({
                "frame_index": frame["frame_index"],
                "confidence": frame["confidence"],
                "is_suspicious": frame["is_suspicious"],
                "face_base64": face_b64,
                "heatmap_base64": heatmap_b64
            })

    timeline_base64 = None
    if "time_confidence_data" in result and result["time_confidence_data"]:
        times = [d["time_sec"] for d in result["time_confidence_data"]]
        confs = [d["confidence"] for d in result["time_confidence_data"]]
        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(times, confs)
        ax.axhline(y=0.5, linestyle='--')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Confidence FAKE")
        ax.set_title("Confidence theo thời gian")
        timeline_base64 = plot_to_base64(fig)
        plt.close(fig)

    return jsonify({
        "prediction": result.get("prediction"),
        "confidence": result.get("confidence"),
        "probability": result.get("probability"),
        "num_faces": result.get("num_faces"),
        "frames_base64": frames_base64,
        "timeline_base64": timeline_base64
    })


# ------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
