# config.py

# --- CẤU HÌNH DỮ LIỆU ---
DATA_DIR = 'processed_data'
BATCH_SIZE = 16 

# Kích thước đầu vào cho Nhánh Khuôn mặt (Face Stream)
FACE_IMG_WIDTH = 320
FACE_IMG_HEIGHT = 320

# Kích thước đầu vào cho Nhánh Ngữ cảnh (Context Stream)
CONTEXT_IMG_WIDTH = 224
CONTEXT_IMG_HEIGHT = 224

# --- CẤU HÌNH HUẤN LUYỆN ---
EPOCHS_WARMUP = 15    
EPOCHS_FINETUNE = 50 
VALIDATION_SPLIT = 0.2

# --- CẤU HÌNH LƯU TRỮ ---
MODEL_OUTPUT_DIR = 'output'
MODEL_NAME = 'two_stream_xception_resnet.h5'