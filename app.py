from flask import Flask, g, render_template, request, jsonify, session, redirect, url_for, make_response
import sqlite3
import os
import io
import base64
import requests
from werkzeug.security import generate_password_hash, check_password_hash
from PIL import Image
from predict import predict_deepfake
import random
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import hashlib

# Fix matplotlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = "2a2fd639618205a5bbbc40f0b64f64d8b8e61c417ea9e7bde08360a15ad8c9ef"

# reCAPTCHA Secret
RECAPTCHA_SECRET = "6LfPWBYsAAAAAHU-CUw4F68N6zyksBQYUe7kM2DB"

# Gmail settings
EMAIL_ADDRESS = "nghoanglam1395@gmail.com"
EMAIL_PASSWORD = "yghqackzlcccnmsw"   # APP PASSWORD

# ---------------------------------------------------------
# SEND EMAIL FUNCTION  ←←← BẠN THIẾU HÀM NÀY
# ---------------------------------------------------------
def send_email(to_email, message):

    msg = MIMEMultipart()
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = to_email
    msg["Subject"] = "Xác minh thiết bị đăng nhập"

    msg.attach(MIMEText(message, "plain"))

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()
        print("Email sent successfully!")
    except Exception as e:
        print("Error sending email:", e)


# ---------------------------------------------------------
# DATABASE FILE
# ---------------------------------------------------------
DATABASE = "deepfake_results.db"

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

# ---------------------------------------------------------
# AUTO CREATE TABLES IF NOT EXISTS
# ---------------------------------------------------------
def init_db():
    conn = sqlite3.connect(DATABASE)
    cur = conn.cursor()

    # USERS TABLE
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL
    );
    """)

    # DEVICES TABLE
    cur.execute("""
    CREATE TABLE IF NOT EXISTS devices (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        device_hash TEXT NOT NULL,
        user_agent TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES users(id)
    );
    """)

    conn.commit()
    conn.close()
    print("Database initialized! Tables are ready.")

# ---------------------------------------------------------
# CREATE DEVICE HASH
# ---------------------------------------------------------
def generate_device_hash(request):
    ua = request.headers.get("User-Agent", "")
    ip = request.remote_addr or "0.0.0.0"
    raw = ua + ip
    return hashlib.sha256(raw.encode()).hexdigest()

# ---------------------------------------------------------
# ROUTES
# ---------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")

# ================= REGISTER =================
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

# ================= LOGIN =================
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        return render_template("login.html")

    email = request.form.get("email")
    password = request.form.get("password")

    user = get_db().execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    if not user or not check_password_hash(user["password"], password):
        return "Sai tài khoản hoặc mật khẩu!"

    # 1. DEVICE HASH
    device_hash = generate_device_hash(request)

    # 2. CHECK DEVICE
    db = get_db()
    device = db.execute(
        "SELECT * FROM devices WHERE user_id = ? AND device_hash = ?",
        (user["id"], device_hash)
    ).fetchone()

    if device:
        session["user_id"] = user["id"]
        resp = make_response(redirect(url_for("index")))
        resp.set_cookie("device_id", device_hash, max_age=60*60*24*365)
        return resp

    # 3. NEW DEVICE → SEND OTP
    otp = str(random.randint(100000, 999999))
    session["pending_user"] = user["id"]
    session["pending_device"] = device_hash
    session["otp"] = otp

    send_email(email, f"Thiết bị mới phát hiện. Mã OTP của bạn: {otp}")

    return render_template("verify_device.html", message="Thiết bị mới! Nhập OTP để tiếp tục.")

# ================= VERIFY NEW DEVICE =================
@app.route("/verify-device", methods=["POST"])
def verify_device():
    user_otp = request.form.get("otp")
    real_otp = session.get("otp")

    if user_otp != real_otp:
        return render_template("verify_device.html", message="OTP sai!")

    user_id = session.get("pending_user")
    device_hash = session.get("pending_device")

    db = get_db()
    db.execute(
        "INSERT INTO devices (user_id, device_hash, user_agent) VALUES (?, ?, ?)",
        (user_id, device_hash, request.headers.get("User-Agent"))
    )
    db.commit()

    session.pop("otp", None)
    session.pop("pending_user", None)
    session.pop("pending_device", None)

    session["user_id"] = user_id
    resp = make_response(redirect(url_for("index")))
    resp.set_cookie("device_id", device_hash, max_age=60*60*24*365)

    return resp
@app.route("/resend-otp")
def resend_otp():
    user_id = session.get("pending_user")
    device_hash = session.get("pending_device")

    if not user_id or not device_hash:
        return redirect(url_for("login"))

    # Lấy email user
    db = get_db()
    user = db.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()

    if not user:
        return redirect(url_for("login"))

    # Tạo OTP mới
    otp = str(random.randint(100000, 999999))
    session["otp"] = otp

    # Gửi mail
    send_email(user["email"], f"Mã OTP mới của bạn: {otp}")

    return render_template("verify_device.html",
                           message="OTP mới đã được gửi lại email!")


# ================= LOGOUT =================
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ---------------------------------------------------------
# RUN
# ---------------------------------------------------------
if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=5000, debug=True)
