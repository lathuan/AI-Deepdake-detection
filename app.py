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
import re
import uuid
from datetime import datetime


# OAuth
from authlib.integrations.flask_client import OAuth

# Fix matplotlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

app = Flask(__name__, static_folder='static', template_folder='templates')
app.secret_key = "2a2fd639618205a5bbbc40f0b64f64d8b8e61c417ea9e7bde08360a15ad8c9ef"

# reCAPTCHA Secret
RECAPTCHA_SECRET = "6LdZ9B8sAAAAAMk8Cg-9NYrS2qi_SVH4SnkUPiMq"  # Key mới

# Gmail settings
EMAIL_ADDRESS = "webdeepfake@gmail.com"
EMAIL_PASSWORD = "acql muop kmgv qmqu"   # APP PASSWORD

# Upload folder
UPLOAD_FOLDER = "examples_test_videos"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Global variables for history and temp storage
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
        'save_all_settings': 'Save All Settings',
        'summary': 'Summary',
        'no_suspicious_features': 'No suspicious features detected.',
        'face_frame_alt': 'Detected face in frame',
        'heatmap_alt': 'Heatmap for frame'
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
        'save_all_settings': 'Lưu Tất Cả Cài Đặt',
        'summary': 'Tóm tắt',
        'no_suspicious_features': 'Không phát hiện dấu hiệu bất thường rõ ràng.',
        'face_frame_alt': 'Khuôn mặt phát hiện trong frame',
        'heatmap_alt': 'Bản đồ nhiệt cho frame'
    }
}

app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = False

# THÊM MỚI: Helper function for base64 conversion
def pil_image_to_base64(pil_image):
    if pil_image is None:
        return ""  # Trả về empty nếu không có image
    buf = io.BytesIO()
    pil_image.save(buf, format='JPEG', quality=85)  # Chất lượng tốt, nhỏ file
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# Database functions (comment out nếu không cần, nhưng giữ để tương thích)
def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect('deepfake.db')
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(error):
    if hasattr(g, 'db'):
        g.db.close()

def init_db():
    with app.app_context():
        db = get_db()
        db.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                level INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
            CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
        """)
        db.commit()

# =========================================================
# SEND EMAIL FUNCTION
# =========================================================
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
        server.sendmail(EMAIL_ADDRESS, to_email, msg.as_string())
        server.quit()
        return True
    except Exception as e:
        print(f"Email error: {e}")
        return False

# =========================================================
# ROUTES (Placeholder - Thêm code login/register/history/settings từ gốc nếu có)
# =========================================================
@app.route("/")
def index():
    lang = session.get('lang', 'en')
    t = TRANSLATIONS[lang]
    theme = session.get('theme', 'dark')
    return render_template('index.html', t=t, theme=theme)

# Thêm routes khác nếu cần (e.g., /login, /history, /settings) - Copy từ code gốc

# =========================================================
# VIDEO UPLOAD AND ANALYSIS
# =========================================================
@app.route("/upload", methods=["POST"])
def upload_video():
    global history, temp_storage
    
    print("=== UPLOAD DEBUG START ===")
    print("Request files:", dict(request.files))
    
    if "video" not in request.files: 
        print("ERROR: No video file in request")
        return jsonify({"error": "No file uploaded"}), 400
    
    video_file = request.files["video"]
    print("Video file object:", video_file)
    print("Video filename:", video_file.filename)
    print("Video content type:", video_file.content_type)
    print("Video content length:", video_file.content_length)
    
    if video_file.filename == "": 
        print("ERROR: Empty filename")
        return jsonify({"error": "Empty filename"}), 400
    
    # Kiểm tra định dạng file
    allowed_extensions = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
    file_extension = video_file.filename.rsplit('.', 1)[1].lower() if '.' in video_file.filename else ''
    
    print("File extension:", file_extension)
    
    if file_extension not in allowed_extensions:
        print(f"ERROR: Invalid file format: {file_extension}")
        return jsonify({"error": f"Invalid file format. Supported: {', '.join(allowed_extensions)}"}), 400
    
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
    filename = video_file.filename
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{timestamp}_{video_file.filename}")

    print("Save path:", save_path)
    print("Upload folder exists:", os.path.exists(app.config["UPLOAD_FOLDER"]))
    print("Upload folder path:", os.path.abspath(app.config["UPLOAD_FOLDER"]))

    try:
        video_file.save(save_path)
        print("File saved successfully")
        
        # Kiểm tra file tồn tại và có kích thước
        if os.path.exists(save_path):
            file_size = os.path.getsize(save_path)
            print(f"File size: {file_size} bytes")
            
            if file_size == 0:
                print("ERROR: File is empty")
                return jsonify({"error": "Uploaded file is empty"}), 400
        else:
            print("ERROR: File not saved properly")
            return jsonify({"error": "Failed to save uploaded video"}), 500

    except Exception as e:
        print(f"ERROR saving file: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Failed to save uploaded video: {str(e)}"}), 500

    try:
        print("Starting deepfake prediction...")
        result = predict_deepfake(save_path)
        print("Prediction result:", result)

        if result.get("error"):
            print("Prediction error:", result["error"])
            session.pop('last_result', None) 
            return jsonify({"error": result["error"]}), 500

        # SỬA: Xử lý frames với explanations (thêm features, summary, risk_level, num_suspicious_features)
        frames_web = []
        for frame in result.get("frames_for_web", []):
            face_b64 = pil_image_to_base64(frame.get("face_image")) 
            heatmap_b64 = pil_image_to_base64(frame.get("heatmap_overlay"))
            
            # THÊM MỚI: Include explanations từ analyzer
            features = frame.get("features", [])  # List of dicts: name, description, severity, confidence
            summary = frame.get("summary", "")    # Text summary từ _create_summary
            risk_level = frame.get("risk_level", "low")
            num_suspicious = frame.get("num_suspicious_features", 0)
            
            # Fallback nếu không có suspicious features: Luôn thêm ít nhất 1 neutral để hiển thị
            if not features:
                features = [{
                    'name': 'Không có dấu hiệu bất thường',
                    'description': 'Tất cả đặc điểm đều tự nhiên, phù hợp với video real.',
                    'severity': 'low',
                    'confidence': 0.0
                }]
                summary = "✅ Video có vẻ REAL. Không phát hiện dấu hiệu bất thường rõ ràng."
            
            frames_web.append({
                "frame_index": frame.get("frame_index"),
                "confidence": frame.get("confidence"),
                "is_suspicious": frame.get("is_suspicious", False),
                "face_base64": face_b64,
                "heatmap_base64": heatmap_b64,
                
                # THÊM MỚI: Gửi explanations về JS để render
                "features": features,
                "summary": summary,
                "risk_level": risk_level,
                "num_suspicious_features": num_suspicious
            })

        response_data = {
            "prediction": result.get("overall_prediction"),
            "confidence": result.get("overall_confidence", result.get("confidence")),
            "frames_base64": frames_web,  # Giờ có đầy đủ features!
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

        print("Upload completed successfully")
        return jsonify(response_data)

    except Exception as e:
        session.pop('last_result', None)
        print(f"Prediction failed due to error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    finally:
        try:
            if os.path.exists(save_path):
                os.remove(save_path)
                print("Temporary file cleaned up")
        except Exception as e:
            print(f"Warning: Failed to delete uploaded video {save_path}: {e}")
    
    print("=== UPLOAD DEBUG END ===")

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
# LOGOUT
# =========================================================
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# =========================================================
# RUN APP (Bỏ qua init_db)
# =========================================================
if __name__ == "__main__":
    # init_db()  # Bỏ qua DB init
    os.makedirs("examples_test_videos", exist_ok=True)
    app.run(host="127.0.0.1", port=5000, debug=False)