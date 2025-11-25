from flask import Flask, g, render_template, request, jsonify, session, redirect, url_for, make_response, flash
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
import time

# OAuth
from authlib.integrations.flask_client import OAuth

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
# SEND EMAIL FUNCTION
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

app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = False

# ---------------------------------------------------------
# GOOGLE LOGIN FIX
# ---------------------------------------------------------
oauth = OAuth(app)
oauth.register(
    name="google",
    client_id="680496606730-7l1fqt20cdtv5gkoaldaunj55r40jul2.apps.googleusercontent.com",
    client_secret="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",
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    userinfo_endpoint='https://openidconnect.googleapis.com/v1/userinfo',
    client_kwargs={"scope": "openid email profile"}
)


# ---------------------------
# Database helper
# ---------------------------
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DATABASE, check_same_thread=False)
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(exception):
    db = g.pop("db", None)
    if db:
        db.close()

def init_db():
    conn = sqlite3.connect(DATABASE)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL
    );
    """)

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

# ---------------------------
# Device / OTP
# ---------------------------
def generate_device_hash(request):
    ua = request.headers.get("User-Agent", "")
    ip = request.remote_addr or "0.0.0.0"
    raw = ua + "|" + ip
    return hashlib.sha256(raw.encode()).hexdigest()

def is_logged_in():
    return "user_id" in session


# =========================================================
# HOME
# =========================================================
@app.route("/")
def index():
    return render_template("index.html", logged=is_logged_in())


# =========================================================
# REGISTER — FIXED: FLASH + VALIDATE + REDIRECT LOGIN
# =========================================================
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "GET":
        return render_template("register.html")

    name = request.form.get("name")
    email = request.form.get("email")
    password = request.form.get("password")

    # Password mạnh
    import re
    if len(password) < 8 or not re.search(r"[A-Z]", password) or not re.search(r"[^A-Za-z0-9]", password):
        flash("Password phải ≥ 8 ký tự, có 1 IN HOA và 1 ký tự đặc biệt!", "error")
        return redirect("/register")

    hashed_pw = generate_password_hash(password)
    db = get_db()

    exists = db.execute("SELECT id FROM users WHERE email=?", (email,)).fetchone()
    if exists:
        flash("Email đã tồn tại!", "error")
        return redirect("/register")

    db.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
               (name, email, hashed_pw))
    db.commit()

    # Thông báo đăng ký thành công
    flash("Đăng ký thành công! Vui lòng đăng nhập.", "success")
    return redirect("/login")


# =========================================================
# LOGIN — FIXED FLASH + SHOW SUCCESS MESSAGE
# =========================================================
import requests

RECAPTCHA_SECRET = "6LfPWBYsAAAAAHU-CUw4F68N6zyksBQYUe7kM2DB"  # SECRET KEY đúng

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        return render_template("login.html")

    # === KIỂM TRA reCAPTCHA ===
    recaptcha_response = request.form.get("g-recaptcha-response")

    if not recaptcha_response:
        return "Vui lòng xác nhận reCAPTCHA!"

    verify_url = "https://www.google.com/recaptcha/api/siteverify"
    data = {
        "secret": RECAPTCHA_SECRET,
        "response": recaptcha_response,
        "remoteip": request.remote_addr
    }

    recaptcha_verify = requests.post(verify_url, data=data).json()

    if not recaptcha_verify.get("success"):
        return "reCAPTCHA không hợp lệ!"

    # === LOGIN SAU KHI PASS CAPTCHA ===

    email = request.form.get("email")
    password = request.form.get("password")

    db = get_db()
    user = db.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()

    if not user or not check_password_hash(user["password"], password):
        return "Sai tài khoản hoặc mật khẩu!"

    # Device verification
    device_hash = generate_device_hash(request)
    device = db.execute(
        "SELECT * FROM devices WHERE user_id=? AND device_hash=?",
        (user["id"], device_hash)
    ).fetchone()

    if device:
        session["user_id"] = user["id"]
        resp = make_response(redirect(url_for("index")))
        resp.set_cookie("device_id", device_hash)
        return resp

    otp = str(random.randint(100000, 999999))
    session["pending_user"] = user["id"]
    session["pending_device"] = device_hash
    session["otp"] = otp
    session["last_otp_time"] = time.time()

    send_email(user["email"], f"Mã OTP: {otp}")
    return render_template("verify_device.html", message="Thiết bị mới! Nhập OTP")


@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "GET":
        return render_template("forgot_password.html")

    email = request.form.get("email")
    db = get_db()
    user = db.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()

    if not user:
        flash("Email không tồn tại!", "error")
        return redirect("/forgot-password")

    # Tạo token đặt lại mật khẩu
    token = hashlib.sha256(f"{email}{time.time()}".encode()).hexdigest()

    # Lưu token tạm vào session
    session["reset_email"] = email
    session["reset_token"] = token

    # Link đặt lại mật khẩu
    reset_link = f"http://127.0.0.1:5000/reset-password/{token}"

    send_email(email, f"Hãy nhấn vào link để đặt lại mật khẩu: {reset_link}")

    flash("Link đặt lại mật khẩu đã được gửi qua email!", "success")
    return redirect("/login")
import re



@app.route("/reset-password/<token>", methods=["GET", "POST"])
def reset_password(token):
    if token != session.get("reset_token"):
        return "Token không hợp lệ hoặc đã hết hạn!"

    if request.method == "GET":
        return render_template("reset_password.html")

    new_password = request.form.get("password")

    # KIỂM TRA MẬT KHẨU
    if len(new_password) < 7:
        flash("Mật khẩu phải ít nhất 7 ký tự!", "danger")
        return render_template("reset_password.html")

    if not re.search(r"[A-Z]", new_password):
        flash("Mật khẩu phải có ít nhất 1 chữ hoa!", "danger")
        return render_template("reset_password.html")

    if not re.search(r"[a-z]", new_password):
        flash("Mật khẩu phải có ít nhất 1 chữ thường!", "danger")
        return render_template("reset_password.html")

    if not re.search(r"[0-9]", new_password):
        flash("Mật khẩu phải có ít nhất 1 số!", "danger")
        return render_template("reset_password.html")

    if not re.search(r"[\W_]", new_password):
        flash("Mật khẩu phải có ít nhất 1 ký tự đặc biệt!", "danger")
        return render_template("reset_password.html")

    # Cập nhật mật khẩu
    hashed = generate_password_hash(new_password)
    email = session.get("reset_email")

    db = get_db()
    db.execute("UPDATE users SET password=? WHERE email=?", (hashed, email))
    db.commit()

    session.pop("reset_email", None)
    session.pop("reset_token", None)

    flash("Đặt lại mật khẩu thành công! Hãy đăng nhập.", "success")
    return redirect("/login")


@app.route("/account")
def account():
    if not is_logged_in():
        return redirect(url_for("login"))

    db = get_db()
    user = db.execute("SELECT * FROM users WHERE id=?", (session["user_id"],)).fetchone()

    return render_template("account.html", user=user)

# =========================================================
# GOOGLE LOGIN
# =========================================================
@app.route("/auth/google")
def auth_google():
    redirect_uri = url_for("auth_google_callback", _external=True)
    return oauth.google.authorize_redirect(redirect_uri)

@app.route("/auth/google/callback")
def auth_google_callback():
    token = oauth.google.authorize_access_token()
    resp = oauth.google.get("https://openidconnect.googleapis.com/v1/userinfo")
    userinfo = resp.json()

    email = userinfo.get("email")
    name = userinfo.get("name")

    if not email:
        return "Không thể lấy email từ Google!"

    db = get_db()
    user = db.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()

    if not user:
        db.execute(
            "INSERT INTO users (username, email, password) VALUES (?, ?, '')",
            (name, email)
        )
        db.commit()
        user = db.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()

    device_hash = generate_device_hash(request)
    device = db.execute(
        "SELECT * FROM devices WHERE user_id=? AND device_hash=?",
        (user["id"], device_hash)
    ).fetchone()

    if device:
        session["user_id"] = user["id"]
        resp = make_response(redirect(url_for("index")))
        resp.set_cookie("device_id", device_hash)
        return resp

    otp = str(random.randint(100000, 999999))
    session["pending_user"] = user["id"]
    session["pending_device"] = device_hash
    session["otp"] = otp
    session["last_otp_time"] = time.time()

    send_email(email, f"Mã OTP: {otp}")
    flash("Thiết bị mới! Nhập OTP", "info")
    return render_template("verify_device.html")


# =========================================================
# VERIFY DEVICE
# =========================================================
@app.route("/verify-device", methods=["POST"])
def verify_device():
    user_otp = request.form.get("otp")
    real_otp = session.get("otp")

    if not real_otp:
        flash("OTP đã hết hạn!", "error")
        return render_template("verify_device.html")

    if user_otp != real_otp:
        flash("OTP sai!", "error")
        return render_template("verify_device.html")

    user_id = session.get("pending_user")
    device_hash = session.get("pending_device")

    db = get_db()
    db.execute(
        "INSERT INTO devices (user_id, device_hash, user_agent) VALUES (?, ?, ?)",
        (user_id, device_hash, request.headers.get("User-Agent"))
    )
    db.commit()

    session.clear()
    session["user_id"] = user_id

    resp = make_response(redirect(url_for("index")))
    resp.set_cookie("device_id", device_hash)

    return resp


# =========================================================
# RESEND OTP
# =========================================================
@app.route("/resend-otp")
def resend_otp():
    last = session.get("last_otp_time", 0)
    now = time.time()

    if now - last < 60:
        wait = 60 - int(now - last)
        flash(f"Chờ {wait} giây để gửi lại OTP!", "error")
        return render_template("verify_device.html")

    otp = str(random.randint(100000, 999999))
    session["otp"] = otp
    session["last_otp_time"] = now

    db = get_db()
    user = db.execute("SELECT * FROM users WHERE id=?", (session["pending_user"],)).fetchone()

    send_email(user["email"], f"OTP mới: {otp}")
    flash("OTP mới đã gửi!", "success")
    return render_template("verify_device.html")


# =========================================================
# LOGOUT
# =========================================================
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# =========================================================
# VIDEO UPLOAD
# =========================================================
@app.route("/upload", methods=["POST"])
def upload_video():
    if not is_logged_in():
        return jsonify({"error": "Unauthorized"}), 403

    if "video" not in request.files:
        return jsonify({"error": "No file"}), 400

    video = request.files["video"]
    path = os.path.join("examples_test_videos", video.filename)
    video.save(path)

    result = predict_deepfake(path)
    os.remove(path)

    return jsonify(result)


# =========================================================
# RUN APP
# =========================================================
if __name__ == "__main__":
    init_db()
    os.makedirs("examples_test_videos", exist_ok=True)
    app.run(host="0.0.0.0", port=5000, debug=True)
