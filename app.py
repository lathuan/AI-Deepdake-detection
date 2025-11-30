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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)