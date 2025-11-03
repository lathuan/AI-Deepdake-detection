# config.py - PHIÊN BẢN CẢI THIỆN

# --- LEARNING RATE CẤU HÌNH ---
LEARNING_RATE_WARMUP = 1e-4    # Tốc độ học tập cho Pha 1: Warm-up
LEARNING_RATE_FINETUNE = 1e-5  # Tốc độ học tập cho Pha 2: Fine-tuning

# --- CẤU HÌNH DỮ LIỆU ---
DATA_DIR = 'processed_data_small'
BATCH_SIZE = 16
VALIDATION_SPLIT = 0.2  # 80% train, 20% validation

# Kích thước đầu vào cho Nhánh Khuôn mặt (Face Stream)
FACE_IMG_WIDTH = 320
FACE_IMG_HEIGHT = 320

# Kích thước đầu vào cho Nhánh Ngữ cảnh (Context Stream)
CONTEXT_IMG_WIDTH = 224
CONTEXT_IMG_HEIGHT = 224

# --- CẤU HÌNH HUẤN LUYỆN ---
EPOCHS_WARMUP = 15      # Số epochs cho giai đoạn warm-up
EPOCHS_FINETUNE = 25    # Số epochs cho giai đoạn fine-tuning

# --- CẤU HÌNH CALLBACKS ---
PATIENCE_WARMUP = 7     # EarlyStopping patience cho warm-up
PATIENCE_FINETUNE = 15  # EarlyStopping patience cho fine-tuning
LR_REDUCE_FACTOR = 0.5  # Factor để giảm learning rate
LR_REDUCE_PATIENCE = 3  # Patience cho ReduceLROnPlateau

# --- CẤU HÌNH MÔ HÌNH ---
DROPOUT_RATE_STREAM = 0.4   # Dropout rate sau các Dense layers trong stream
DROPOUT_RATE_COMBINED = 0.3 # Dropout rate sau kết hợp hai stream
DENSE_UNITS_1 = 128         # Dense layer 1 (Face & Context output)
DENSE_UNITS_2 = 64          # Dense layer 2 (combined)
DENSE_UNITS_3 = 32          # Dense layer 3 (combined) - THÊMMỚI

# --- CẤU HÌNH ĐẦURATO ---
MODEL_OUTPUT_DIR = 'models'
MODEL_NAME = 'two_stream_deepfake_model.h5'
FINAL_MODEL_NAME = 'final_two_stream_deepfake_model.h5'

# --- CẤU HÌNH FINE-TUNING ---
UNFREEZE_LAYERS_XCEPTION = 50   # Số lớp cuối của Xception để mở khóa
UNFREEZE_LAYERS_RESNET = 50     # Số lớp cuối của ResNet50 để mở khóa