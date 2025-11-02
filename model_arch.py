# model_arch.py

import tensorflow as tf
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Concatenate, BatchNormalization
from tensorflow.keras.applications import Xception, ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# --- HÀM TẠO MÔ HÌNH TWO-STREAM ---
def create_two_stream_model(face_input_shape, context_input_shape):
    # 1. Nhánh Khuôn mặt (Face Stream) - Dùng Xception
    face_input = Input(shape=face_input_shape, name='face_input')
    # base model: Xception (QUAN TRỌNG: Đặt tên cho base model là 'xception')
    face_base = Xception(weights='imagenet', include_top=False, input_tensor=face_input, name='xception')
    
    # Đóng băng tất cả các lớp của Xception cho Warm-up
    for layer in face_base.layers:
        layer.trainable = False
        
    x = face_base.output
    x = GlobalAveragePooling2D()(x)
    face_output = Dense(128, activation='relu')(x)
    
    # 2. Nhánh Ngữ cảnh (Context Stream) - Dùng ResNet50
    context_input = Input(shape=context_input_shape, name='context_input')
    # base model: ResNet50 (QUAN TRỌNG: Đặt tên cho base model là 'resnet50')
    context_base = ResNet50(weights='imagenet', include_top=False, input_tensor=context_input, name='resnet50')
    
    # Đóng băng tất cả các lớp của ResNet50 cho Warm-up
    for layer in context_base.layers:
        layer.trainable = False
        
    y = context_base.output
    y = GlobalAveragePooling2D()(y)
    context_output = Dense(128, activation='relu')(y)
    
    # 3. Kết hợp và Phân loại
    combined = Concatenate()([face_output, context_output])
    combined = BatchNormalization()(combined)
    combined = Dense(64, activation='relu')(combined)
    output = Dense(2, activation='softmax')(combined) # 2 lớp: fake/real
    
    model = Model(inputs=[face_input, context_input], outputs=output)
    
    return model


# --- HÀM TINH CHỈNH (FINE-TUNING) - ĐÃ SỬA LỖI TÌM LỚP NỀN ---
def fine_tune_two_stream_model(model, learning_rate_finetune):
    
    # 1. Tìm lớp nền bằng Tên đã đặt
    try:
        # Lấy base model bằng tên đã đặt trong hàm create_two_stream_model()
        xception_base = model.get_layer('xception') 
        resnet_base = model.get_layer('resnet50')
    except ValueError:
        print("LỖI: KHÔNG THỂ TINH CHỈNH. Kiểm tra lại tên lớp ('xception'/'resnet50') trong create_two_stream_model().")
        # Biên dịch lại mô hình với LR mới (mặc dù các lớp nền vẫn đóng băng)
        model.compile(
            optimizer=Adam(learning_rate=learning_rate_finetune),
            loss=model.loss,
            metrics=model.metrics
        )
        return model 

    # 2. Bắt đầu Mở khóa (Unfreeze)
    
    # Mở khóa 50 lớp cuối của Xception
    for layer in xception_base.layers[-50:]: 
        # Giữ BatchNormalization (BN) bị đóng băng là best practice
        if not isinstance(layer, BatchNormalization):
            layer.trainable = True
        
    # Mở khóa 50 lớp cuối của ResNet50
    for layer in resnet_base.layers[-50:]: 
        # Giữ BN bị đóng băng
        if not isinstance(layer, BatchNormalization):
            layer.trainable = True

    print(f"--- ĐÃ MỞ KHÓA 50 lớp cuối của Xception và ResNet50. ---")

    # 3. Biên dịch lại (Recompile) với Tốc độ học tập mới
    model.compile(
        optimizer=Adam(learning_rate=learning_rate_finetune),
        loss=model.loss,
        metrics=model.metrics
    )
    
    return model