
# config.py - PHIÊN BẢN CẢI TIẾN V2

# --- LEARNING RATE CẤU HÌNH ---
LEARNING_RATE_WARMUP = 5e-4    # Tốc độ học tập cho Pha 1: Warm-up (tăng từ 1e-4)
LEARNING_RATE_FINETUNE = 1e-5  # Tốc độ học tập cho Pha 2: Fine-tuning

# --- CẤU HÌNH DỮ LIỆU ---
DATA_DIR = 'processed_data'
BATCH_SIZE = 32  # Tăng từ 16 để cải thiện gradient estimation
VALIDATION_SPLIT = 0.2  # 80% train, 20% validation

# Kích thước đầu vào cho Nhánh Khuôn mặt (Face Stream)
FACE_IMG_WIDTH = 320
FACE_IMG_HEIGHT = 320

# Kích thước đầu vào cho Nhánh Ngữ cảnh (Context Stream)
CONTEXT_IMG_WIDTH = 224
CONTEXT_IMG_HEIGHT = 224

# --- CẤU HÌNH HUẤN LUYỆN ---
EPOCHS_WARMUP = 20      # Tăng từ 15
EPOCHS_FINETUNE = 35    # Tăng từ 25

# --- CẤU HÌNH CALLBACKS ---
PATIENCE_WARMUP = 10     # Tăng từ 7
PATIENCE_FINETUNE = 18   # Tăng từ 15
LR_REDUCE_FACTOR = 0.5   # Factor để giảm learning rate
LR_REDUCE_PATIENCE = 5   # Tăng từ 3 (chờ lâu hơn trước khi reduce)

# --- CẤU HÌNH MÔ HÌNH ---
DROPOUT_RATE_STREAM = 0.5    # Tăng từ 0.4
DROPOUT_RATE_COMBINED = 0.4  # Tăng từ 0.3
DENSE_UNITS_1 = 256         # Tăng từ 128 - cho phép học đặc trưng phức tạp hơn
DENSE_UNITS_2 = 128          # Tăng từ 64
DENSE_UNITS_3 = 64           # Tăng từ 32

# --- CẤU HÌNH ĐẦU RA ---
MODEL_OUTPUT_DIR = 'models'
MODEL_NAME = 'best_deepfake_model.keras'  # ✓ SỬA: Chỉ lưu 1 model tốt nhất

# --- CẤU HÌNH FINE-TUNING ---
UNFREEZE_LAYERS_XCEPTION = 80   # Tăng từ 50 - mở khóa nhiều lớp hơn
UNFREEZE_LAYERS_RESNET = 80     # Tăng từ 50

# --- CẤU HÌNH MỚI: XỬ LÝ CLASS IMBALANCE ---
USE_CLASS_WEIGHTS = True  # Sử dụng class weights
USE_FOCAL_LOSS = False    # Dùng Focal Loss nếu có tensorflow_addons
FOCAL_LOSS_ALPHA = 0.25   # Alpha parameter cho Focal Loss
FOCAL_LOSS_GAMMA = 2.0    # Gamma parameter cho Focal Loss

# --- CẤU HÌNH MỚI: DATA AUGMENTATION NÂNG CAO ---
USE_ADVANCED_AUGMENTATION = True  # Bật CutMix, MixUp
CUTMIX_PROB = 0.5         # Xác suất áp dụng CutMix
MIXUP_PROB = 0.3           # Xác suất áp dụng MixUp
MIXUP_ALPHA = 0.2          # Alpha parameter cho MixUp

# --- CẤU HÌNH MỚI: FACE DETECTION ---
USE_MEDIAPIPE_FACE = True  # Dùng MediaPipe thay vì Haar Cascade
MIN_DETECTION_CONFIDENCE = 0.5

# --- CẤU HÌNH MỚI: K-FOLD CROSS VALIDATION ---
USE_KFOLD = False          # Bật K-Fold (khuyến nghị cho dataset nhỏ)
N_SPLITS = 5               # Số fold

# --- CẤU HÌNH MỚI: REGULARIZATION ---
L2_REGULARIZATION = 1e-4   # L2 regularization cho dense layers