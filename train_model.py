# train_model.py (FINAL - LOẠI BỎ CLASS_WEIGHT VÀ HÀM LOSS TÙY CHỈNH)

import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
# BỎ: from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from tensorflow.keras.losses import CategoricalCrossentropy

# 1. IMPORT CẤU HÌNH VÀ KIẾN TRÚC
from config import *
from model_arch import create_two_stream_model, fine_tune_two_stream_model


# --- HÀM TẠO DATA GENERATOR CHO MÔ HÌNH HAI NHÁNH ---
def get_two_stream_generator(data_dir, target_size_face, target_size_context, batch_size, subset, validation_split):
    datagen = ImageDataGenerator(
        rescale=1./255, 
        rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, 
        shear_range=0.1, zoom_range=0.1, horizontal_flip=True,
        validation_split=VALIDATION_SPLIT 
    )

    face_gen = datagen.flow_from_directory(
        data_dir, target_size=target_size_face, batch_size=batch_size,
        class_mode='categorical', subset=subset, seed=42
    )

    context_gen = datagen.flow_from_directory(
        data_dir, target_size=target_size_context, batch_size=batch_size,
        class_mode='categorical', subset=subset, seed=42
    )
    
    total_samples = face_gen.n 
    
    def two_stream_generator():
        while True:
            X_face = face_gen.__next__() 
            X_context = context_gen.__next__()
            yield ({'face_input': X_face[0], 'context_input': X_context[0]}, X_face[1])

    return two_stream_generator(), total_samples


# --- HÀM CHÍNH ĐỂ HUẤN LUYỆN (ĐÃ LOẠI BỎ class_weights) ---
def train_model():
    # 1. TẠO DATA GENERATORS VÀ TÍNH TOÁN
    train_gen, train_samples = get_two_stream_generator(
        data_dir=DATA_DIR, target_size_face=(FACE_IMG_WIDTH, FACE_IMG_HEIGHT),
        target_size_context=(CONTEXT_IMG_WIDTH, CONTEXT_IMG_HEIGHT),
        batch_size=BATCH_SIZE, subset='training', validation_split=VALIDATION_SPLIT
    )
    
    val_gen, val_samples = get_two_stream_generator(
        data_dir=DATA_DIR, target_size_face=(FACE_IMG_WIDTH, FACE_IMG_HEIGHT),
        target_size_context=(CONTEXT_IMG_WIDTH, CONTEXT_IMG_HEIGHT),
        batch_size=BATCH_SIZE, subset='validation', validation_split=VALIDATION_SPLIT
    )

    train_steps = train_samples // BATCH_SIZE
    val_steps = val_samples // BATCH_SIZE
    
    # BỎ CÁC DÒNG TÍNH TOÁN TRỌNG SỐ LỚP
    
    # In ra thông tin cơ bản (thay thế cho output tính trọng số lớp cũ)
    print(f"Found {train_samples} images belonging to 2 classes. (Training)")
    print(f"Found {val_samples} images belonging to 2 classes. (Validation)")

    
    # 2. TẠO VÀ HUẤN LUYỆN MÔ HÌNH (GIAI ĐOẠN WARM-UP)
    print("\n[PHASE 1] Bắt đầu Huấn luyện Warm-up (Các lớp nền bị đóng băng)...")
    
    model = create_two_stream_model(
        face_input_shape=(FACE_IMG_WIDTH, FACE_IMG_HEIGHT, 3),
        context_input_shape=(CONTEXT_IMG_WIDTH, CONTEXT_IMG_HEIGHT, 3)
    )

    # COMPILE VỚI HÀM LOSS CHUẨN
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_WARMUP),
        loss=CategoricalCrossentropy(), # Dùng Loss chuẩn
        metrics=['accuracy']
    )

    warmup_callbacks = [
        EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1),
    ]

    if train_steps > 0:
        model.fit(
            train_gen, steps_per_epoch=train_steps, epochs=EPOCHS_WARMUP,
            validation_data=val_gen, validation_steps=val_steps,
            callbacks=warmup_callbacks
            # KHÔNG TRUYỀN class_weight VÀO fit()
        )
    else:
        print("LỖI: Số lượng bước huấn luyện (train_steps) bằng 0. Kiểm tra lại dữ liệu và BATCH_SIZE.")
        return 
    
    # 3. TINH CHỈNH (FINE-TUNING)
    print("\n[PHASE 2] Bắt đầu Tinh chỉnh (Mở khóa các lớp cuối)...")

    model = fine_tune_two_stream_model(model, LEARNING_RATE_FINETUNE) 

    finetune_callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1),
        ModelCheckpoint(os.path.join(MODEL_OUTPUT_DIR, MODEL_NAME), monitor='val_loss', save_best_only=True)
    ]
    
    model.fit(
        train_gen, steps_per_epoch=train_steps, epochs=EPOCHS_FINETUNE,
        validation_data=val_gen, validation_steps=val_steps,
        callbacks=finetune_callbacks
        # KHÔNG TRUYỀN class_weight VÀO fit()
    )

    # 4. LƯU MÔ HÌNH CUỐI CÙNG
    final_model_path = os.path.join(MODEL_OUTPUT_DIR, f'final_{MODEL_NAME}')
    model.save(final_model_path)
    print(f"\n--- Hoàn thành huấn luyện. Mô hình cuối cùng được lưu tại: {final_model_path} ---")


if __name__ == '__main__':
    if not os.path.exists(MODEL_OUTPUT_DIR):
        os.makedirs(MODEL_OUTPUT_DIR)
        
    train_model()