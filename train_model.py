# train_model.py - PHIÊN BẢN CẢI THIỆN

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau, 
                                       ModelCheckpoint, TensorBoard)
from sklearn.utils.class_weight import compute_class_weight

# Import config và model
from config import *
from model_arch import (create_two_stream_model, fine_tune_two_stream_model, 
                       compile_model, print_model_summary)


# --- HÀM TẠO DATA GENERATOR CHO MÔ HÌNH HAI NHÁNH (CẢI THIỆN) ---
def get_two_stream_generator(data_dir, target_size_face, target_size_context, 
                            batch_size, subset, validation_split):
    """
    Tạo data generator cho hai nhánh xử lý ảnh kích thước khác nhau
    
    Args:
        data_dir: Đường dẫn thư mục chứa dữ liệu
        target_size_face: Kích thước ảnh cho Face stream (H, W)
        target_size_context: Kích thước ảnh cho Context stream (H, W)
        batch_size: Batch size
        subset: 'training' hoặc 'validation'
        validation_split: Tỷ lệ validation split
    
    Returns:
        Generator và số lượng samples
    """
    
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        validation_split=validation_split
    )
    
    face_gen = datagen.flow_from_directory(
        data_dir,
        target_size=target_size_face,
        batch_size=batch_size,
        class_mode='categorical',
        subset=subset,
        seed=42
    )
    
    context_gen = datagen.flow_from_directory(
        data_dir,
        target_size=target_size_context,
        batch_size=batch_size,
        class_mode='categorical',
        subset=subset,
        seed=42
    )
    
    total_samples = face_gen.n
    
    def two_stream_generator():
        while True:
            X_face = face_gen.__next__()
            X_context = context_gen.__next__()
            yield ({'face_input': X_face[0], 'context_input': X_context[0]}, X_face[1])
    
    return two_stream_generator(), total_samples


# --- HÀM TÍNH CLASS WEIGHTS CHO IMBALANCED DATA (SỬA - KHÔNG TRUYỀN class_weight) ---
def calculate_class_weights(data_dir, subset='training', validation_split=0.2):
    """
    Tính class weights để xử lý mất cân bằng dữ liệu
    
    Returns:
        Dict: {class_index: weight}
    """
    datagen = ImageDataGenerator(validation_split=validation_split)
    
    gen = datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=1,
        class_mode='categorical',
        subset=subset,
        shuffle=False
    )
    
    # Lấy labels của tất cả samples
    all_labels = []
    for _ in range(gen.n):
        _, labels = gen.__next__()
        all_labels.append(np.argmax(labels, axis=1)[0])
    
    all_labels = np.array(all_labels)
    
    # Tính class weights
    class_weights = compute_class_weight('balanced',
                                        classes=np.array([0, 1]),
                                        y=all_labels)
    
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
    
    print(f"📊 Class Weights (xử lý imbalanced data):")
    print(f"   Class 0 (Real): {class_weight_dict[0]:.4f}")
    print(f"   Class 1 (Deepfake): {class_weight_dict[1]:.4f}")
    
    return class_weight_dict


# --- HÀM CHÍNH ĐỂ HUẤN LUYỆN ---
def train_model(use_class_weights=True, use_focal_loss=False):
    """
    Huấn luyện mô hình hai giai đoạn: Warm-up và Fine-tuning
    
    Args:
        use_class_weights: Sử dụng class weights cho imbalanced data
        use_focal_loss: Sử dụng Focal Loss (requires tensorflow_addons)
    """
    
    # ===== BƯỚC 1: CHUẨN BỊ DỮ LIỆU =====
    print("\n" + "="*80)
    print("📂 CHUẨN BỊ DỮ LIỆU")
    print("="*80)
    
    train_gen, train_samples = get_two_stream_generator(
        data_dir=DATA_DIR,
        target_size_face=(FACE_IMG_WIDTH, FACE_IMG_HEIGHT),
        target_size_context=(CONTEXT_IMG_WIDTH, CONTEXT_IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        subset='training',
        validation_split=VALIDATION_SPLIT
    )
    
    val_gen, val_samples = get_two_stream_generator(
        data_dir=DATA_DIR,
        target_size_face=(FACE_IMG_WIDTH, FACE_IMG_HEIGHT),
        target_size_context=(CONTEXT_IMG_WIDTH, CONTEXT_IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        subset='validation',
        validation_split=VALIDATION_SPLIT
    )
    
    train_steps = train_samples // BATCH_SIZE
    val_steps = val_samples // BATCH_SIZE
    
    print(f"✓ Training samples: {train_samples} ({train_steps} steps)")
    print(f"✓ Validation samples: {val_samples} ({val_steps} steps)")
    
    if train_steps == 0:
        print("❌ LỖI: train_steps = 0. Kiểm tra lại dữ liệu và BATCH_SIZE")
        return
    
    # Tính class weights nếu cần
    class_weight_dict = None
    if use_class_weights:
        print("\n📊 Tính toán Class Weights...")
        class_weight_dict = calculate_class_weights(DATA_DIR, subset='training', 
                                                   validation_split=VALIDATION_SPLIT)
    
    
    # ===== BƯỚC 2: TẠO MÔ HÌNH =====
    print("\n" + "="*80)
    print("🏗 TẠO MÔ HÌNH TWO-STREAM")
    print("="*80)
    
    model = create_two_stream_model(
        face_input_shape=(FACE_IMG_WIDTH, FACE_IMG_HEIGHT, 3),
        context_input_shape=(CONTEXT_IMG_WIDTH, CONTEXT_IMG_HEIGHT, 3),
        dropout_stream=DROPOUT_RATE_STREAM,
        dropout_combined=DROPOUT_RATE_COMBINED,
        dense_1=DENSE_UNITS_1,
        dense_2=DENSE_UNITS_2,
        dense_3=DENSE_UNITS_3
    )
    
    # In thông tin mô hình
    print_model_summary(model, verbose=False)
    
    # Compile mô hình
    model = compile_model(model, LEARNING_RATE_WARMUP, use_focal_loss=use_focal_loss)
    
    
    # ===== BƯỚC 3: HỘI TỤ (WARMUP) =====
    print("\n" + "="*80)
    print("🔥 GIAI ĐOẠN 1: WARM-UP (Lớp nền bị đóng băng)")
    print("="*80)
    print(f"Learning Rate: {LEARNING_RATE_WARMUP}")
    print(f"Epochs: {EPOCHS_WARMUP}")
    
    warmup_callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=PATIENCE_WARMUP,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=LR_REDUCE_FACTOR,
            patience=LR_REDUCE_PATIENCE,
            min_lr=1e-6,
            verbose=1
        ),
        TensorBoard(
            log_dir=os.path.join(MODEL_OUTPUT_DIR, 'logs_warmup'),
            histogram_freq=1
        )
    ]
    
    # SỬA: Loại bỏ class_weight trong fit()
    history_warmup = model.fit(
        train_gen,
        steps_per_epoch=train_steps,
        epochs=EPOCHS_WARMUP,
        validation_data=val_gen,
        validation_steps=val_steps,
        callbacks=warmup_callbacks,
        verbose=1
        # KHÔNG TRUYỀN class_weight ĐÂY TRUYỀN
    )
    
    print("\n✓ Hoàn thành giai đoạn Warm-up")
    
    
    # ===== BƯỚC 4: TINH CHỈNH (FINE-TUNING) =====
    print("\n" + "="*80)
    print("🔓 GIAI ĐOẠN 2: FINE-TUNING (Mở khóa lớp cuối)")
    print("="*80)
    print(f"Learning Rate: {LEARNING_RATE_FINETUNE}")
    print(f"Epochs: {EPOCHS_FINETUNE}")
    
    model = fine_tune_two_stream_model(
        model,
        LEARNING_RATE_FINETUNE,
        unfreeze_xception=UNFREEZE_LAYERS_XCEPTION,
        unfreeze_resnet=UNFREEZE_LAYERS_RESNET
    )
    
    finetune_callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=PATIENCE_FINETUNE,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=LR_REDUCE_FACTOR,
            patience=LR_REDUCE_PATIENCE,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=os.path.join(MODEL_OUTPUT_DIR, MODEL_NAME),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        TensorBoard(
            log_dir=os.path.join(MODEL_OUTPUT_DIR, 'logs_finetune'),
            histogram_freq=1
        )
    ]
    
    # SỬA: Loại bỏ class_weight trong fit()
    history_finetune = model.fit(
        train_gen,
        steps_per_epoch=train_steps,
        epochs=EPOCHS_FINETUNE,
        validation_data=val_gen,
        validation_steps=val_steps,
        callbacks=finetune_callbacks,
        verbose=1
        # KHÔNG TRUYỀN class_weight ĐÂY TRUYỀN
    )
    
    print("\n✓ Hoàn thành giai đoạn Fine-tuning")
    
    
    # ===== BƯỚC 5: LƯU MÔ HÌNH =====
    print("\n" + "="*80)
    print("💾 LƯU MÔ HÌNH")
    print("="*80)
    
    final_model_path = os.path.join(MODEL_OUTPUT_DIR, FINAL_MODEL_NAME)
    model.save(final_model_path)
    print(f"✓ Mô hình cuối cùng: {final_model_path}")
    
    best_model_path = os.path.join(MODEL_OUTPUT_DIR, MODEL_NAME)
    print(f"✓ Mô hình tốt nhất: {best_model_path}")
    
    print("\n" + "="*80)
    print("🎉 HOÀN THÀNH HUẤN LUYỆN")
    print("="*80 + "\n")
    
    return model, history_warmup, history_finetune


if __name__ == '__main__':
    # Tạo thư mục output nếu chưa có
    if not os.path.exists(MODEL_OUTPUT_DIR):
        os.makedirs(MODEL_OUTPUT_DIR)
        print(f"✓ Tạo thư mục: {MODEL_OUTPUT_DIR}")
    
    # Bắt đầu huấn luyện
    # use_focal_loss=True nếu bạn có tensorflow_addons cài
    model, hist_warmup, hist_finetune = train_model(use_class_weights=True, 
                                                    use_focal_loss=False)