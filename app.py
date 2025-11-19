
#app.py
from flask import Flask, render_template, request, jsonify
import os
import io
import base64
import matplotlib.pyplot as plt
from PIL import Image
from predict import predict_deepfake

app = Flask(__name__)

UPLOAD_FOLDER = "examples_test_videos"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/")
def index():
    return render_template("index.html")

def pil_image_to_base64(img):
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

def plot_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

@app.route("/upload", methods=["POST"])
def upload_video():
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

    # Frame + heatmap
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

    # Timeline
    timeline_base64 = None
    if "time_confidence_data" in result and result["time_confidence_data"]:
        times = [d["time_sec"] for d in result["time_confidence_data"]]
        confs = [d["confidence"] for d in result["time_confidence_data"]]
        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(times, confs, color="blue", linewidth=2)
        ax.axhline(y=0.5, color='red', linestyle='--')
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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
