from flask import Flask, g, render_template, request, jsonify, session, redirect, url_for
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
# SEND EMAIL OTP
# ---------------------------------------------------------
def send_email(to_email, content):
    msg = MIMEMultipart()
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = to_email
    msg["Subject"] = "OTP Code"

    msg.attach(MIMEText(content, "plain"))

    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
    server.send_message(msg)
    server.quit()


# ---------------------------------------------------------
# DATABASE
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


# ---------------------------------------------------------
# UPLOAD FOLDER
# ---------------------------------------------------------
UPLOAD_FOLDER = "examples_test_videos"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


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
    captcha_response = request.form.get("g-recaptcha-response")

    verify_url = "https://www.google.com/recaptcha/api/siteverify"
    payload = {"secret": RECAPTCHA_SECRET, "response": captcha_response}
    captcha_verify = requests.post(verify_url, data=payload).json()

    if not captcha_verify.get("success"):
        return "Captcha không hợp lệ!"

    user = get_db().execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()

    if not user or not check_password_hash(user["password"], password):
        return "Sai tài khoản hoặc mật khẩu!"

    session["user_id"] = user["id"]
    session["email"] = user["email"]

    return redirect(url_for("index"))


# ================= LOGOUT =================
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ==================================================
#       FORGOT PASSWORD + OTP
# ==================================================
@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    # GET → Chỉ hiện ô email
    if request.method == "GET":
        return render_template("forgot_verify.html", show_otp=False, message=None, email=None)

    action = request.form.get("action")

    # -------------------------------------------------
    # SEND OTP
    # -------------------------------------------------
    if action == "send_otp":
        email = request.form.get("email")

        if not email:
            return render_template("forgot_verify.html", show_otp=False, message="Vui lòng nhập email!", email=None)

        user = get_db().execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        if not user:
            return render_template("forgot_verify.html", show_otp=False, message="Email không tồn tại!", email=None)

        otp = str(random.randint(100000, 999999))
        session["reset_email"] = email
        session["reset_otp"] = otp

        send_email(email, f"Your OTP code is: {otp}")

        return render_template("forgot_verify.html", show_otp=True, message="Đã gửi OTP!", email=email)

    # -------------------------------------------------
    # VERIFY OTP
    # -------------------------------------------------
    if action == "verify_otp":
        user_otp = request.form.get("otp")

        if user_otp == session.get("reset_otp"):
            return redirect("/reset-password")

        return render_template("forgot_verify.html",
                               show_otp=True,
                               message="OTP không đúng!",
                               email=session.get("reset_email"))

    return redirect("/forgot-password")


# ==================================================
#          RESET PASSWORD
# ==================================================
@app.route("/reset-password", methods=["GET", "POST"])
def reset_password():
    if request.method == "GET":
        return render_template("reset_password.html")

    new_pw = request.form.get("password")
    email = session.get("reset_email")

    if not email:
        return "Phiên làm việc hết hạn, vui lòng thử lại!"

    hashed_pw = generate_password_hash(new_pw)

    db = get_db()
    db.execute("UPDATE users SET password = ? WHERE email = ?", (hashed_pw, email))
    db.commit()

    session.pop("reset_email", None)
    session.pop("reset_otp", None)

    return "Đổi mật khẩu thành công! <a href='/login'>Đăng nhập</a>"


# ---------------------------------------------------------
# RUN
# ---------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
