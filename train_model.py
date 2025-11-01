# train_model.py

import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import pandas as pd

# 1. IMPORT CẤU HÌNH VÀ KIẾN TRÚC
from config import *
from model_arch import create_two_stream_model, fine_tune_two_stream_model


# --- HÀM TẠO DATA GENERATOR CHO MÔ HÌNH HAI NHÁNH (ĐÃ SỬA LỖI) ---
def get_two_stream_generator(data_dir, target_size_face, target_size_context, batch_size, subset, validation_split):
    datagen = ImageDataGenerator(
        rescale=1./255, # Chuẩn hóa
        rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, 
        shear_range=0.1, zoom_range=0.1, horizontal_flip=True,
        validation_split=validation_split
    )

    # 1. Tạo Generator cho nhánh Khuôn mặt (để lấy số lượng mẫu .n)
    face_gen = datagen.flow_from_directory(
        data_dir, target_size=target_size_face, batch_size=batch_size,
        class_mode='categorical', subset=subset, seed=42
    )

    # 2. Tạo Generator cho nhánh Ngữ cảnh
    context_gen = datagen.flow_from_directory(
        data_dir, target_size=target_size_context, batch_size=batch_size,
        class_mode='categorical', subset=subset, seed=42
    )
    
    # 3. Lấy tổng số mẫu (.n) từ generator của Keras (rất quan trọng)
    total_samples = face_gen.n 
    
    # 4. Định nghĩa Generator tùy chỉnh để gộp input
    def two_stream_generator():
        while True:
            X_face = face_gen.next()
            X_context = context_gen.next()
            
            # Trả về dictionary cho hai đầu vào của mô hình và nhãn
            yield ({'face_input': X_face[0], 'context_input': X_context[0]}, X_face[1])

    # 5. Trả về generator VÀ tổng số mẫu
    return two_stream_generator(), total_samples


# --- HÀM TÍNH TRỌNG SỐ LỚP (Giữ nguyên) ---
def get_class_weights(data_dir, validation_split):
    # Tạo generator cơ sở CHỈ để lấy các nhãn (classes) của tập huấn luyện
    temp_gen_base = ImageDataGenerator(validation_split=validation_split).flow_from_directory(
        data_dir, 
        target_size=(FACE_IMG_WIDTH, FACE_IMG_HEIGHT), 
        batch_size=1, 
        subset='training', 
        class_mode='categorical', 
        shuffle=False 
    )
    
    classes_labels = temp_gen_base.classes
    unique_classes = np.unique(classes_labels)
    
    weights = compute_class_weight('balanced', classes=unique_classes, y=classes_labels)
    class_weights = dict(zip(unique_classes, weights))
    
    print(f"Indices: {temp_gen_base.class_indices}")
    print(f"Trọng số lớp được tính: {class_weights}")
    return class_weights


# --- HÀM CHÍNH ĐỂ HUẤN LUYỆN (ĐÃ SỬA LỖI) ---
def train_model():
    # 1. TẠO DATA GENERATORS VÀ TÍNH TOÁN
    # Sửa lỗi: Hàm get_two_stream_generator giờ trả về 2 giá trị
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

    # Tính toán steps_per_epoch bằng cách sử dụng số lượng mẫu đã lấy
    train_steps = train_samples // BATCH_SIZE
    val_steps = val_samples // BATCH_SIZE
    class_weights = get_class_weights(DATA_DIR, VALIDATION_SPLIT)
    
    
    # 2. TẠO VÀ HUẤN LUYỆN MÔ HÌNH (GIAI ĐOẠN WARM-UP)
    print("\n[PHASE 1] Bắt đầu Huấn luyện Warm-up (Các lớp nền bị đóng băng)...")
    
    model = create_two_stream_model(
        face_input_shape=(FACE_IMG_WIDTH, FACE_IMG_HEIGHT, 3),
        context_input_shape=(CONTEXT_IMG_WIDTH, CONTEXT_IMG_HEIGHT, 3)
    )

    warmup_callbacks = [
        EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1),
    ]

    # Kiểm tra nếu train_steps > 0 trước khi fit
    if train_steps > 0:
        model.fit(
            train_gen, steps_per_epoch=train_steps, epochs=EPOCHS_WARMUP,
            validation_data=val_gen, validation_steps=val_steps,
            class_weight=class_weights, callbacks=warmup_callbacks
        )
    else:
        print("LỖI: Số lượng bước huấn luyện (train_steps) bằng 0. Kiểm tra lại dữ liệu và BATCH_SIZE.")
        return # Dừng nếu không có dữ liệu để train
    
    # 3. TINH CHỈNH (FINE-TUNING)
    print("\n[PHASE 2] Bắt đầu Tinh chỉnh (Mở khóa các lớp cuối)...")

    model = fine_tune_two_stream_model(model) 

    finetune_callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1),
        ModelCheckpoint(os.path.join(MODEL_OUTPUT_DIR, MODEL_NAME), monitor='val_loss', save_best_only=True)
    ]
    
    model.fit(
        train_gen, steps_per_epoch=train_steps, epochs=EPOCHS_FINETUNE,
        validation_data=val_gen, validation_steps=val_steps,
        class_weight=class_weights, callbacks=finetune_callbacks
    )

    # 4. LƯU MÔ HÌNH CUỐI CÙNG
    final_model_path = os.path.join(MODEL_OUTPUT_DIR, f'final_{MODEL_NAME}')
    model.save(final_model_path)
    print(f"\n--- Hoàn thành huấn luyện. Mô hình cuối cùng được lưu tại: {final_model_path} ---")


if __name__ == '__main__':
    if not os.path.exists(MODEL_OUTPUT_DIR):
        os.makedirs(MODEL_OUTPUT_DIR)
        
    train_model()