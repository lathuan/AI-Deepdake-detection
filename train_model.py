# train_model.py

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.utils import class_weight
import numpy as np
import os
import pandas as pd
from config import (
    DATA_FOLDERS, NUM_CLASSES, IMG_WIDTH, IMG_HEIGHT, CHANNELS, 
    BATCH_SIZE, EPOCHS, VALIDATION_SPLIT, SEED, MODEL_FILE, 
    HISTORY_FILE, OUTPUT_DIR, PROCESSED_DATA_DIR
)
from model_arch import create_mesonet


# Đảm bảo thư mục output tồn tại
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def get_data_generator():
    """Tạo ImageDataGenerator để tải và tăng cường dữ liệu."""
    
    # Tải dữ liệu từ thư mục processed_data/ và chia thành train/val
    datagen = ImageDataGenerator(
        rescale=1./255,                 # Chuẩn hóa ảnh
        validation_split=VALIDATION_SPLIT # Tỉ lệ chia validation
        # Bạn có thể thêm các tham số data augmentation khác tại đây
    )
    
    train_generator = datagen.flow_from_directory(
        PROCESSED_DATA_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        color_mode="rgb",
        batch_size=BATCH_SIZE,
        class_mode='categorical',       # Cần categorical cho đầu ra softmax
        subset='training',
        seed=SEED
    )
    
    validation_generator = datagen.flow_from_directory(
        PROCESSED_DATA_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        color_mode="rgb",
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        seed=SEED
    )
    
    return train_generator, validation_generator

def train_model():
    train_gen, val_gen = get_data_generator()
    
    # 1. Tính Class Weights (Quan trọng vì dữ liệu REAL/FAKE có thể không cân bằng)
    # Lấy các nhãn số nguyên từ generator
    labels = train_gen.classes
    
    weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(labels),
        y=labels
    )
    class_weights = dict(zip(np.unique(labels), weights))
    print(f"Trọng số lớp (Class Weights): {class_weights}")

    # 2. Tạo mô hình và Compile
    model = create_mesonet(IMG_WIDTH, IMG_HEIGHT, CHANNELS, NUM_CLASSES)
    
    # Sử dụng categorical_crossentropy vì đầu ra là 2 neurons (REAL, FAKE) với softmax
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) 

    # 3. Định nghĩa Callbacks
    checkpoint = ModelCheckpoint(
        os.path.join(OUTPUT_DIR, MODEL_FILE),
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    early_stop = EarlyStopping(monitor='val_loss', patience=5)
    
    # 4. Huấn luyện
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=[checkpoint, early_stop],
        class_weight=class_weights,
        verbose=1
    )
    
    # 5. Lưu lịch sử huấn luyện
    pd.DataFrame(history.history).to_csv(os.path.join(OUTPUT_DIR, HISTORY_FILE), index=False)
    print(f"Huấn luyện hoàn tất. Mô hình tốt nhất được lưu tại {os.path.join(OUTPUT_DIR, MODEL_FILE)}")

if __name__ == '__main__':
    train_model()