from flask import Flask, render_template, request, jsonify
import os
from predict import predict_deepfake

app = Flask(__name__)

UPLOAD_FOLDER = "examples_test_videos"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ==========================
#  TRANG CHÍNH
# ==========================
@app.route("/")
def index():
    return render_template("index.html")

# ==========================
#  UPLOAD + PREDICT
# ==========================
@app.route("/upload", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    video_file = request.files["video"]
    if video_file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    save_path = os.path.join(app.config["UPLOAD_FOLDER"], video_file.filename)
    video_file.save(save_path)

    # Gọi AI
    result = predict_deepfake(save_path)

    # Xóa file tạm
    os.remove(save_path)

    return jsonify(result)

# ==========================
#  RUN FLASK
# ==========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
