
from flask import Flask, render_template, request, jsonify, session
import os, io, base64
from datetime import datetime
import uuid 
try:
    from PIL import Image
except ImportError:
    print("WARNING: PIL/Pillow is required. Please run: pip install Pillow")
    exit()

try:
    from predict import predict_deepfake 
except ImportError:
    def predict_deepfake(video_path):
        import time
        timeline_img = Image.new('RGB', (600, 150), color = 'blue')
        buf = io.BytesIO()
        timeline_img.save(buf, format='PNG') 
        timeline_b64_string = base64.b64encode(buf.getvalue()).decode()
        
        face_img = Image.new('RGB', (100, 100), color = 'red')
        heatmap_img = Image.new('RGB', (100, 100), color = 'yellow')
        
        time.sleep(2) 
        return {
            "overall_prediction": "FAKE",
            "overall_confidence": 0.92,
            "timeline_base64": timeline_b64_string,
            "frames_for_web": [
                {"frame_index": 10, "confidence": 0.88, "is_suspicious": True, "face_image": face_img, "heatmap_overlay": heatmap_img},
                {"frame_index": 50, "confidence": 0.95, "is_suspicious": True, "face_image": face_img, "heatmap_overlay": heatmap_img}
            ]
        }
    print("INFO: Using mock 'predict_deepfake' function.")


app = Flask(__name__)
app.secret_key = 'super_secret_key_change_me' 
UPLOAD_FOLDER = "examples_test_videos"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

history = [] 
temp_storage = {} 

# Translation dictionaries
TRANSLATIONS = {
    'en': {
        'dashboard': 'Dashboard',
        'history': 'History',
        'settings': 'Settings',
        'logout': 'Logout',
        'level': 'Level',
        'uid': 'UID',
        'hero_title': 'AI Deepfake Detection',
        'tagline': 'Upload your videos and detect deepfakes instantly',
        'upload_section': 'Upload Video for Analysis',
        'select_video': 'Select Video',
        'no_file': 'No file selected',
        'analyze': 'Analyze',
        'supported_formats': 'Supported formats: MP4, AVI, MOV',
        'save_to_history': 'Save video {} to history?',
        'save': 'Save',
        'discard': 'Discard',
        'analysis_result': 'Analysis Result',
        'prediction': 'Prediction',
        'confidence': 'Confidence',
        'suspicious_timeline': 'Suspicious Timeline',
        'suspicious_frames': 'Suspicious Frames',
        'detected_face': 'Detected Face',
        'heatmap': 'Heatmap',
        'analysis_history': 'Analysis History',
        'review_results': 'Review all saved deepfake analysis results.',
        'saved_results': 'Saved Results',
        'entries': 'entries',
        'analyzed': 'Analyzed',
        'no_timeline': 'No timeline data available for this entry.',
        'no_history': 'No history entries found. Run an analysis on the Dashboard and click Save.',
        'settings_title': 'SETTINGS',
        'settings_tagline': 'Manage your account and application preferences',
        'notifications': 'Notifications',
        'notifications_desc': 'Enable/disable notifications when a video analysis completes.',
        'appearance': 'Appearance',
        'appearance_desc': 'Choose theme and language.',
        'theme': 'Theme',
        'dark': 'Dark',
        'light': 'Light',
        'language': 'Language',
        'video_analysis': 'Video Analysis',
        'video_analysis_desc': 'Set number of suspicious frames to display after analysis.',
        'save_settings': 'Save All Settings',
        'settings_saved': 'Settings saved!',
        'clear_all': 'Clear All',
        'delete': 'Delete',
        'confirm_delete': 'Confirm Delete',
        'confirm_clear_all': 'Clear All History',
        'delete_item_confirm': 'Are you sure you want to delete ?',
        'clear_all_warning': 'This will permanently delete all your analysis history. This action cannot be undone.',
        'yes_delete': 'Yes, Delete',
        'yes_clear_all': 'Yes, Clear All',
        'cancel': 'Cancel',
        'no_history_desc': 'Run an analysis on the Dashboard and save the results to see them here.',
        'apply': 'Apply',
        'save_all_settings': 'Save All Settings'
    },
    'vi': {
        'dashboard': 'Bảng điều khiển',
        'history': 'Lịch sử',
        'settings': 'Cài đặt',
        'logout': 'Đăng xuất',
        'level': 'Cấp độ',
        'uid': 'UID',
        'hero_title': 'Phát Hiện Deepfake AI',
        'tagline': 'Tải video lên và phát hiện deepfake ngay lập tức',
        'upload_section': 'Tải Video Lên Để Phân Tích',
        'select_video': 'Chọn Video',
        'no_file': 'Chưa chọn file',
        'analyze': 'Phân Tích',
        'supported_formats': 'Định dạng hỗ trợ: MP4, AVI, MOV',
        'save_to_history': 'Lưu video {} vào lịch sử?',
        'save': 'Lưu',
        'discard': 'Bỏ qua',
        'analysis_result': 'Kết Quả Phân Tích',
        'prediction': 'Dự đoán',
        'confidence': 'Độ tin cậy',
        'suspicious_timeline': 'Dòng thời gian đáng ngờ',
        'suspicious_frames': 'Khung hình đáng ngờ',
        'detected_face': 'Khuôn mặt phát hiện',
        'heatmap': 'Bản đồ nhiệt',
        'analysis_history': 'Lịch Sử Phân Tích',
        'review_results': 'Xem lại tất cả kết quả phân tích deepfake đã lưu.',
        'saved_results': 'Kết Quả Đã Lưu',
        'entries': 'mục',
        'analyzed': 'Đã phân tích',
        'no_timeline': 'Không có dữ liệu dòng thời gian cho mục này.',
        'no_history': 'Không tìm thấy mục lịch sử nào. Hãy chạy phân tích trên Bảng điều khiển và nhấn Lưu.',
        'settings_title': 'CÀI ĐẶT',
        'settings_tagline': 'Quản lý tài khoản và tùy chọn ứng dụng',
        'notifications': 'Thông báo',
        'notifications_desc': 'Bật/tắt thông báo khi phân tích video hoàn tất.',
        'appearance': 'Giao diện',
        'appearance_desc': 'Chọn chủ đề và ngôn ngữ.',
        'theme': 'Chủ đề',
        'dark': 'Tối',
        'light': 'Sáng',
        'language': 'Ngôn ngữ',
        'video_analysis': 'Phân Tích Video',
        'video_analysis_desc': 'Đặt số lượng khung hình đáng ngờ hiển thị sau khi phân tích.',
        'save_settings': 'Lưu Tất Cả Cài Đặt',
        'settings_saved': 'Đã lưu cài đặt!',
        'clear_all': 'Xóa Tất Cả',
        'delete': 'Xóa',
        'confirm_delete': 'Xác Nhận Xóa',
        'confirm_clear_all': 'Xóa Toàn Bộ Lịch Sử',
        'delete_item_confirm': 'Bạn có chắc muốn xóa ?',
        'clear_all_warning': 'Hành động này sẽ xóa vĩnh viễn toàn bộ lịch sử phân tích. Không thể hoàn tác.',
        'yes_delete': 'Có, Xóa',
        'yes_clear_all': 'Có, Xóa Tất Cả',
        'cancel': 'Hủy',
        'no_history_desc': 'Chạy phân tích trên Bảng điều khiển và lưu kết quả để xem ở đây.',
        'apply': 'Áp dụng',
        'save_all_settings': 'Lưu Tất Cả Cài Đặt'
    }
}

def pil_image_to_base64(img):
    if img is None:
        return None
    if not isinstance(img, Image.Image):
        return None
    buf = io.BytesIO()
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img.save(buf, format='JPEG', quality=70) 
    return base64.b64encode(buf.getvalue()).decode()

def get_user_data():
    return {
        "username": "Deepfake Analyst",
        "tagline": "AI Model V5.0 Ready",
        "icon_url": "https://via.placeholder.com/60/007BFF/FFFFFF?text=AN",
        "level": "Pro",
        "uid": "DFA-001"
    }

def get_translations(lang='en'):
    return TRANSLATIONS.get(lang, TRANSLATIONS['en'])

@app.route("/")
def index():
    user_data = get_user_data()
    language = session.get('language', 'en')
    theme = session.get('theme', 'dark')
    translations = get_translations(language)
    last_result = session.get('last_result')
    return render_template("index.html", user=user_data, last_result=last_result, 
                         t=translations, language=language, theme=theme)

@app.route("/history")
def history_page():
    user_data = get_user_data()
    language = session.get('language', 'en')
    theme = session.get('theme', 'dark')
    translations = get_translations(language)
    return render_template("history.html", user=user_data, history=list(reversed(history)),
                         t=translations, language=language, theme=theme)

@app.route("/settings")
def settings_page():
    user_data = get_user_data()
    language = session.get('language', 'en')
    theme = session.get('theme', 'dark')
    translations = get_translations(language)
    return render_template("settings.html", user=user_data, 
                         t=translations, language=language, theme=theme)

@app.route("/update_settings", methods=["POST"])
def update_settings():
    data = request.get_json()
    theme = data.get('theme', 'dark')
    language = data.get('language', 'en')
    
    session['theme'] = theme
    session['language'] = language
    
    # Đảm bảo session được lưu
    session.modified = True
    
    return jsonify({"status": "success", "message": "Settings updated successfully"})


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
        flash("Vui lòng xác nhận reCAPTCHA!", "error")
        return redirect(url_for("login"))

    verify_url = "https://www.google.com/recaptcha/api/siteverify"
    data = {
        "secret": RECAPTCHA_SECRET,
        "response": recaptcha_response,
        "remoteip": request.remote_addr
    }

    recaptcha_verify = requests.post(verify_url, data=data).json()

    if not recaptcha_verify.get("success"):
        flash("reCAPTCHA không hợp lệ!", "error")
        return redirect(url_for("login"))

    # === LOGIN SAU KHI PASS CAPTCHA ===

    email = request.form.get("email")
    password = request.form.get("password")

    db = get_db()
    user = db.execute(
        "SELECT * FROM users WHERE email = ?", 
        (email,)
    ).fetchone()

    if not user or not check_password_hash(user["password"], password):
        flash("Sai tài khoản hoặc mật khẩu!", "error")
        return redirect(url_for("login"))

    # === KIỂM TRA THIẾT BỊ ===

    device_hash = generate_device_hash(request)
    device = db.execute(
        "SELECT * FROM devices WHERE user_id=? AND device_hash=?",
        (user["id"], device_hash)
    ).fetchone()

    # ---- THIẾT BỊ ĐÃ TỒN TẠI → LOGIN NGAY ----
    if device:
        session["user_id"] = user["id"]
        resp = make_response(redirect(url_for("index")))
        resp.set_cookie("device_id", device_hash, max_age=86400*30)
        return resp

    # ---- THIẾT BỊ MỚI → GỬI OTP ----
    otp = str(random.randint(100000, 999999))

    session["pending_user"] = user["id"]
    session["pending_device"] = device_hash
    session["otp"] = otp
    session["otp_expire"] = time.time() + 60      # OTP chỉ tồn tại 60 giây
    session["last_otp_time"] = time.time()


    send_email(user["email"], f"Mã OTP xác minh thiết bị của bạn là: {otp}")

    return render_template("verify_device.html", message="Thiết bị mới! Vui lòng nhập OTP để xác minh.")

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
    try:
        token = oauth.google.authorize_access_token()
    except Exception as e:
        return f"OAuth failed: {e}"

    # LẤY THÔNG TIN USER ĐÚNG CÁCH
    userinfo = oauth.google.userinfo()

    if userinfo is None:
        return "Không thể lấy thông tin Google!"

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

    # KIỂM TRA THIẾT BỊ
    device_hash = generate_device_hash(request)
    device = db.execute(
        "SELECT * FROM devices WHERE user_id=? AND device_hash=?",
        (user["id"], device_hash)
    ).fetchone()

    if device:
        session["user_id"] = user["id"]
        resp = make_response(redirect(url_for("index")))
        resp.set_cookie("device_id", device_hash, max_age=86400 * 30)
        return resp

    # THIẾT BỊ MỚI → GỬI OTP
    otp = str(random.randint(100000, 999999))

    session["pending_user"] = user["id"]
    session["pending_device"] = device_hash
    session["otp"] = otp
    session["otp_expire"] = time.time() + 60
    session["last_otp_time"] = time.time()

    send_email(email, f"Mã OTP của bạn là: {otp}")

    flash("Thiết bị mới! Vui lòng nhập OTP để xác minh.", "info")
    return render_template("verify_device.html")


# =========================================================
# VERIFY DEVICE
# =========================================================
@app.route("/verify-device", methods=["POST"])
def verify_device():
    user_otp = request.form.get("otp")
    real_otp = session.get("otp")

    # Không có OTP trong session
    if not real_otp:
        flash("OTP đã hết hạn! Vui lòng yêu cầu mã mới.", "error")
        return render_template("verify_device.html")

    # Kiểm tra hết hạn 60s
    expire = session.get("otp_expire", 0)
    if time.time() > expire:
        flash("OTP đã hết hạn! Vui lòng yêu cầu mã mới.", "error")
        return render_template("verify_device.html")

    # Sai OTP
    if user_otp != real_otp:
        flash("OTP không đúng!", "error")
        return render_template("verify_device.html")

    # OTP đúng → lưu thiết bị
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
    session["otp_expire"] = time.time() + 60   # OTP mới tồn tại thêm 60s
    session["last_otp_time"] = now

    db = get_db()
    user = db.execute("SELECT * FROM users WHERE id=?", (session["pending_user"],)).fetchone()

    send_email(user["email"], f"OTP mới: {otp}")
    flash("OTP mới đã gửi!", "success")
    return render_template("verify_device.html")

@app.route("/change-password", methods=["GET", "POST"])
def change_password():
    if "user_id" not in session:
        flash("Bạn cần đăng nhập trước!", "error")
        return redirect("/login")

    if request.method == "GET":
        return render_template("change_password.html")

    old_pw = request.form.get("old_password")
    new_pw = request.form.get("new_password")
    confirm_pw = request.form.get("confirm_password")

    # Lấy dữ liệu user từ DB
    db = get_db()
    user = db.execute("SELECT * FROM users WHERE id = ?", (session["user_id"],)).fetchone()

    # Kiểm tra mật khẩu cũ đúng
    if not check_password_hash(user["password"], old_pw):
        flash("Mật khẩu cũ không chính xác!", "error")
        return redirect("/change-password")

    # Kiểm tra mật khẩu mới trùng xác nhận
    if new_pw != confirm_pw:
        flash("Xác nhận mật khẩu không khớp!", "error")
        return redirect("/change-password")

    # *********************
    # KIỂM TRA MẬT KHẨU MẠNH (chỉ cần 7 ký tự)
    # *********************
    import re
    def is_strong(p):
        return (
            len(p) >= 7
            and re.search(r"[A-Z]", p)
            and re.search(r"[a-z]", p)
            and re.search(r"[0-9]", p)
            and re.search(r"[!@#$%^&*(),.?\":{}|<>]", p)
        )

    if not is_strong(new_pw):
        flash("Mật khẩu phải ≥ 7 ký tự và bao gồm: chữ hoa, chữ thường, số, ký tự đặc biệt!", "error")
        return redirect("/change-password")

    # Lưu mật khẩu mới
    hashed = generate_password_hash(new_pw)
    db.execute("UPDATE users SET password = ? WHERE id = ?", (hashed, session["user_id"]))
    db.commit()

    flash("Đổi mật khẩu thành công!", "success")
    return redirect("/account")

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

    global history, temp_storage
    
    if "video" not in request.files: return jsonify({"error": "No file uploaded"}), 400
    video_file = request.files["video"]
    if video_file.filename == "": return jsonify({"error": "Empty filename"}), 400
    
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
    filename = video_file.filename
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{timestamp}_{video_file.filename}")

    try:
        video_file.save(save_path)
    except Exception as e:
        return jsonify({"error": f"Failed to save uploaded video: {str(e)}"}), 500

    try:
        result = predict_deepfake(save_path)

        if result.get("error"):
            session.pop('last_result', None) 
            return jsonify({"error": result["error"]}), 500

        frames_web = []
        for frame in result.get("frames_for_web", []):
            face_b64 = pil_image_to_base64(frame.get("face_image")) 
            heatmap_b64 = pil_image_to_base64(frame.get("heatmap_overlay"))
            frames_web.append({
                "frame_index": frame.get("frame_index"),
                "confidence": frame.get("confidence"),
                "is_suspicious": frame.get("is_suspicious", False),
                "face_base64": face_b64,
                "heatmap_base64": heatmap_b64
            })

        response_data = {
            "prediction": result.get("overall_prediction"),
            "confidence": result.get("overall_confidence", result.get("confidence")),
            "frames_base64": frames_web,
            "timeline_base64": result.get("timeline_base64") 
        }
        
        session['last_result'] = response_data 

        temp_entry = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "filename": filename,
            "prediction": result.get("overall_prediction"),
            "confidence": result.get("overall_confidence", result.get("confidence")),
            "timeline_base64": response_data["timeline_base64"]
        }
        
        temp_id = str(uuid.uuid4())
        temp_storage[temp_id] = temp_entry
        response_data["temp_id"] = temp_id 

        return jsonify(response_data)

    except Exception as e:
        session.pop('last_result', None)
        print(f"Prediction failed due to error: {e}") 
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    finally:
        try:
            if os.path.exists(save_path):
                os.remove(save_path)
        except Exception as e:
            print(f"Warning: Failed to delete uploaded video {save_path}: {e}")

@app.route("/save_history", methods=["POST"])
def save_history():
    global history, temp_storage
    
    data = request.get_json()
    temp_id = data.get("temp_id")
    should_save = data.get("save", False) 

    if temp_id in temp_storage:
        entry_to_save = temp_storage.pop(temp_id)
        
        if should_save:
            history.append(entry_to_save)
            if len(history) > 50:
                history.pop(0)
            return jsonify({"status": "success", "message": "History saved."})
        else:
            return jsonify({"status": "success", "message": "History discarded."})
    else:
        return jsonify({"status": "error", "message": "Invalid or expired temp ID."}), 404

@app.route("/delete_history_item", methods=["POST"])
def delete_history_item():
    global history
    data = request.get_json()
    index = data.get("index")
    
    if index is not None and 0 <= index < len(history):
        deleted_item = history.pop(index)
        return jsonify({"status": "success", "message": "Item deleted successfully"})
    
    return jsonify({"status": "error", "message": "Invalid index"}), 400

@app.route("/clear_all_history", methods=["POST"])
def clear_all_history():
    global history
    history.clear()
    return jsonify({"status": "success", "message": "All history cleared successfully"})

# =========================================================
# RUN APP
# =========================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

