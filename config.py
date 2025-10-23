# config.py

# --- Cấu hình Tập Dữ liệu (Phân loại Nhị phân: REAL/FAKE) ---
# Tên các thư mục chứa ảnh khuôn mặt đã cắt, đặt trong thư mục 'processed_data/'
DATA_FOLDERS = {
    "real": 0,    # Nhãn số 0: Ảnh khuôn mặt thật
    "fake": 1     # Nhãn số 1: Tất cả ảnh deepfake
}

NUM_CLASSES = len(DATA_FOLDERS)  # Sẽ bằng 2

# --- Cấu hình Ảnh Đầu vào ---
IMG_WIDTH = 256
IMG_HEIGHT = 256
CHANNELS = 3

# --- Cấu hình Huấn luyện ---
BATCH_SIZE = 32
EPOCHS = 20  # Bạn có thể điều chỉnh số lượng epoch này
VALIDATION_SPLIT = 0.2
SEED = 42

# --- Cấu hình Tên File Đầu ra ---
MODEL_FILE = 'mesonet_binary_detector.h5' 
HISTORY_FILE = 'mesonet_binary_history.csv' 
OUTPUT_DIR = 'output'
PROCESSED_DATA_DIR = 'processed_data'