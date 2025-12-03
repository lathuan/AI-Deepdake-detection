
# train_model.py - PHIÊN BẢN CẢI TIẾN V3 (FIX class_weight error)

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau, 
                                       ModelCheckpoint, TensorBoard)
from sklearn.utils.class_weight import compute_class_weight

from config import *
from model_arch import (create_two_stream_model, fine_tune_two_stream_model, 
                       compile_model, print_model_summary)


def get_two_stream_generator(data_dir, target_size_face, target_size_context, 
                            batch_size, subset, validation_split):
    """
    Tạo data generator cho hai nhánh với augmentation nâng cao
    ✓ FIX: Generator format chính xác
    """
    
    # Augmentation cấu hình
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=validation_split
    )
    
    face_gen = datagen.flow_from_directory(
        data_dir,
        target_size=target_size_face,
        batch_size=batch_size,
        class_mode='categorical',
        subset=subset,
        seed=42,
        interpolation='bilinear'
    )
    
    context_gen = datagen.flow_from_directory(
        data_dir,
        target_size=target_size_context,
        batch_size=batch_size,
        class_mode='categorical',
        subset=subset,
        seed=42,
        interpolation='bilinear'
    )
    
    total_samples = face_gen.n
    
    def two_stream_generator():
        while True:
            X_face = face_gen.__next__()
            X_context = context_gen.__next__()
            # ✓ FIX: Yield đúng format (dict inputs, labels)
            yield (
                {'face_input': X_face[0], 'context_input': X_context[0]}, 
                X_face[1]
            )
    
    return two_stream_generator(), total_samples


def train_model(use_class_weights=True, use_focal_loss=False):
    """
    Huấn luyện mô hình với hai giai đoạn: Warm-up và Fine-tuning
    ✓ FIX: BỎ class_weight vì generator không hỗ trợ
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
    
    train_steps = max(1, train_samples // BATCH_SIZE)
    val_steps = max(1, val_samples // BATCH_SIZE)
    
    print(f"✓ Training samples: {train_samples} ({train_steps} steps)")
    print(f"✓ Validation samples: {val_samples} ({val_steps} steps)")
    print(f"✓ Batch size: {BATCH_SIZE}")
    
    if train_steps == 0:
        print("❌ LỖI: train_steps = 0. Kiểm tra lại dữ liệu và BATCH_SIZE")
        return
    
    
    # ===== BƯỚC 2: TẠO MÔ HÌNH =====
    print("\n" + "="*80)
    print("🏗️  TẠO MÔ HÌNH TWO-STREAM")
    print("="*80)
    
    model = create_two_stream_model(
        face_input_shape=(FACE_IMG_WIDTH, FACE_IMG_HEIGHT, 3),
        context_input_shape=(CONTEXT_IMG_WIDTH, CONTEXT_IMG_HEIGHT, 3),
        dropout_stream=DROPOUT_RATE_STREAM,
        dropout_combined=DROPOUT_RATE_COMBINED,
        dense_1=DENSE_UNITS_1,
        dense_2=DENSE_UNITS_2,
        dense_3=DENSE_UNITS_3,
        l2_reg=L2_REGULARIZATION
    )
    
    print_model_summary(model, verbose=False)
    
    model = compile_model(model, LEARNING_RATE_WARMUP, use_focal_loss=use_focal_loss)
    
    
    # ===== BƯỚC 3: HỎI TỤ (WARMUP) =====
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
    
    # ✓ FIX: BỎ class_weight vì generator không hỗ trợ
    print("\n📊 Ghi chú: class_weight không được hỗ trợ với custom generator")
    print("   Mô hình sẽ tự cân bằng qua augmentation và dropout")
    
    history_warmup = model.fit(
        train_gen,
        steps_per_epoch=train_steps,
        epochs=EPOCHS_WARMUP,
        validation_data=val_gen,
        validation_steps=val_steps,
        callbacks=warmup_callbacks,
        verbose=1
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
            verbose=1,
            mode='min'
        ),
        TensorBoard(
            log_dir=os.path.join(MODEL_OUTPUT_DIR, 'logs_finetune'),
            histogram_freq=1
        )
    ]
    
    # ✓ FIX: BỎ class_weight
    history_finetune = model.fit(
        train_gen,
        steps_per_epoch=train_steps,
        epochs=EPOCHS_FINETUNE,
        validation_data=val_gen,
        validation_steps=val_steps,
        callbacks=finetune_callbacks,
        verbose=1
    )
    
    print("\n✓ Hoàn thành giai đoạn Fine-tuning")
    
    
    # ===== BƯỚC 5: LƯU MODEL =====
    print("\n" + "="*80)
    print("💾 KẾT QUẢ HỮU LUYỆN")
    print("="*80)
    
    best_model_path = os.path.join(MODEL_OUTPUT_DIR, MODEL_NAME)
    
    print(f"\n✅ Model tốt nhất được lưu tại:")
    print(f"   📁 {best_model_path}")
    
    # Hiển thị thông tin model
    if len(history_warmup.history['val_loss']) > 0:
        print(f"\n📊 Thông tin training:")
        print(f"   ├─ Warmup Val Loss (cuối): {history_warmup.history['val_loss'][-1]:.4f}")
        print(f"   ├─ Warmup Val Accuracy: {history_warmup.history['val_accuracy'][-1]:.4f}")
        print(f"   ├─ Fine-tune Val Loss (cuối): {history_finetune.history['val_loss'][-1]:.4f}")
        print(f"   └─ Fine-tune Val Accuracy: {history_finetune.history['val_accuracy'][-1]:.4f}")
    
    print("\n" + "="*80)
    print("🎉 HOÀN THÀNH HUẤn LUYỆN")
    print("="*80 + "\n")
    
    return model, history_warmup, history_finetune


if __name__ == '__main__':
    # Tạo thư mục output nếu chưa có
    if not os.path.exists(MODEL_OUTPUT_DIR):
        os.makedirs(MODEL_OUTPUT_DIR)
        print(f"✓ Tạo thư mục: {MODEL_OUTPUT_DIR}")
    
    # Bắt đầu huấn luyện
    print("\n" + "="*80)
    print("🚀 BẮT ĐẦU HUẤn LUYỆN DEEPFAKE DETECTION")
    print("="*80)
    
    model, hist_warmup, hist_finetune = train_model(
        use_class_weights=USE_CLASS_WEIGHTS, 
        use_focal_loss=USE_FOCAL_LOSS
    )
    
    print("\n✅ TRAINING COMPLETED SUCCESSFULLY!")