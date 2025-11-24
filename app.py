from flask import Flask, g, render_template, request, jsonify, session, redirect, url_for
import sqlite3
import os
import io
import base64
import requests
from werkzeug.security import generate_password_hash, check_password_hash
from PIL import Image
from predict import predict_deepfake

# ===== MATPLOTLIB FIX =====
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# ===========================


app = Flask(__name__)
app.secret_key = "2a2fd639618205a5bbbc40f0b64f64d8b8e61c417ea9e7bde08360a15ad8c9ef"

# Google reCAPTCHA SECRET KEY (KEY BẠN GỬI TRONG ẢNH)
RECAPTCHA_SECRET = "6LfPWBYsAAAAAHU-CUw4F68N6zyksBQYUe7kM2DB"

# SESSION
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = False

DATABASE = "deepfake_results.db"

# ---------------------- DATABASE ----------------------
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DATABASE)
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(exception):
    db = g.pop("db", None)
    if db:
        db.close()

def init_db():
    db = get_db()
    db.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL
        );
    """)
    db.commit()

if not os.path.exists(DATABASE):
    with app.app_context():
        init_db()

# ---------------------- UPLOAD FOLDER ----------------------
UPLOAD_FOLDER = "examples_test_videos"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ---------------------- ROUTES ----------------------
@app.route("/")
def index():
    return render_template("index.html")

# ===================== REGISTER =====================
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

# ===================== LOGIN =========================
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        return render_template("login.html")

    # Get form data
    email = request.form.get("email")
    password = request.form.get("password")
    captcha_response = request.form.get("g-recaptcha-response")

    # ================= CAPTCHA VERIFY ===================
    verify_url = "https://www.google.com/recaptcha/api/siteverify"
    payload = {
        "secret": RECAPTCHA_SECRET,
        "response": captcha_response
    }

    captcha_verify = requests.post(verify_url, data=payload).json()

    if not captcha_verify.get("success"):
        return "Captcha không hợp lệ! Vui lòng thử lại."

    # ================= CHECK LOGIN ======================
    user = get_db().execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()

    if not user or not check_password_hash(user["password"], password):
        return "Sai tài khoản hoặc mật khẩu!"

    session["user_id"] = user["id"]
    session["email"] = user["email"]

    return redirect(url_for("index"))

# ===================== LOGOUT =========================
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ===================== HELPERS =========================
def pil_to_base64(img):
    buff = io.BytesIO()
    img.save(buff, format="PNG")
    return base64.b64encode(buff.getvalue()).decode()

def fig_to_base64(fig):
    buff = io.BytesIO()
    fig.savefig(buff, format="png")
    buff.seek(0)
    return base64.b64encode(buff.read()).decode()

def is_logged_in():
    return "user_id" in session

# ===================== UPLOAD + PREDICT =====================
@app.route("/upload", methods=["POST"])
def upload_video():
    if not is_logged_in():
        return jsonify({"error": "Unauthorized"}), 403

    if "video" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    video = request.files["video"]
    if video.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    path = os.path.join(app.config["UPLOAD_FOLDER"], video.filename)
    video.save(path)

    result = predict_deepfake(path)
    os.remove(path)

    if "error" in result:
        return jsonify(result), 400

    # FRAME RESULTS
    frames_b64 = []
    if "frames_for_web" in result:
        for f in result["frames_for_web"]:
            frames_b64.append({
                "frame_index": f["frame_index"],
                "confidence": f["confidence"],
                "is_suspicious": f["is_suspicious"],
                "face_base64": pil_to_base64(f["face_image"]),
                "heatmap_base64": pil_to_base64(f["heatmap_overlay"]) if f["heatmap_overlay"] else None
            })

    # TIMELINE GRAPH
    timeline_b64 = None
    if result.get("time_confidence_data"):
        times = [x["time_sec"] for x in result["time_confidence_data"]]
        confs = [x["confidence"] for x in result["time_confidence_data"]]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(times, confs)
        ax.axhline(0.5, ls="--")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Confidence Fake")
        ax.set_title("Confidence theo thời gian")

        timeline_b64 = fig_to_base64(fig)
        plt.close(fig)

    return jsonify({
        "prediction": result.get("prediction"),
        "confidence": result.get("confidence"),
        "probability": result.get("probability"),
        "num_faces": result.get("num_faces"),
        "frames_base64": frames_b64,
        "timeline_base64": timeline_b64
    })

# ===================== RUN APP =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
