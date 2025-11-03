# model_arch.py (FINAL - ĐÃ SỬA LỖI TÌM LỚP NỀN MẠNH MẼ HƠN)

import tensorflow as tf
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Concatenate, BatchNormalization
from tensorflow.keras.applications import Xception, ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


# --- HÀM TẠO MÔ HÌNH TWO-STREAM ---
def create_two_stream_model(face_input_shape, context_input_shape):
    # 1. Nhánh Khuôn mặt (Face Stream) - Dùng Xception
    face_input = Input(shape=face_input_shape, name='face_input')
    # Quan trọng: đặt tên
    face_base = Xception(weights='imagenet', include_top=False, input_tensor=face_input, name='xception')
    
    for layer in face_base.layers:
        layer.trainable = False
        
    x = face_base.output
    x = GlobalAveragePooling2D()(x)
    face_output = Dense(128, activation='relu')(x)
    
    # 2. Nhánh Ngữ cảnh (Context Stream) - Dùng ResNet50
    context_input = Input(shape=context_input_shape, name='context_input')
    # Quan trọng: đặt tên
    context_base = ResNet50(weights='imagenet', include_top=False, input_tensor=context_input, name='resnet50')
    
    for layer in context_base.layers:
        layer.trainable = False
        
    y = context_base.output
    y = GlobalAveragePooling2D()(y)
    context_output = Dense(128, activation='relu')(y)
    
    # 3. Kết hợp và Phân loại
    combined = Concatenate()([face_output, context_output])
    combined = BatchNormalization()(combined)
    combined = Dense(64, activation='relu')(combined)
    output = Dense(2, activation='softmax')(combined) 
    
    model = Model(inputs=[face_input, context_input], outputs=output)
    
    return model


# --- HÀM TINH CHỈNH (FINE-TUNING) - ĐÃ SỬA LỖI TÌM KIẾM MẠNH MẼ HƠN ---
def fine_tune_two_stream_model(model, learning_rate_finetune):
    
    xception_base = None
    resnet_base = None
    
    # CHIẾN LƯỢC MỚI: Dò tìm qua tất cả các lớp của mô hình tổng thể
    # và kiểm tra loại của từng lớp
    for layer in model.layers:
        if isinstance(layer, Xception) and layer.name == 'xception':
             xception_base = layer
        elif isinstance(layer, ResNet50) and layer.name == 'resnet50':
             resnet_base = layer
             
    # CƠ CHẾ DỰ PHÒNG 2: Nếu chưa tìm thấy, cố gắng tìm bằng tên (như lần trước)
    if xception_base is None:
        try:
            xception_base = model.get_layer('xception')
        except ValueError:
            pass
    if resnet_base is None:
        try:
            resnet_base = model.get_layer('resnet50')
        except ValueError:
            pass

    
    # Báo lỗi nếu vẫn không tìm thấy và trả về mô hình cũ
    if xception_base is None or resnet_base is None:
         print("LỖI: KHÔNG THỂ TINH CHỈNH. Thất bại khi tìm lớp nền Xception/ResNet50.")
         
         # ĐỂ TRÁNH LỖI TypeError: Mean.update_state, chúng ta compile lại
         # với Loss và Metrics TƯỜNG MINH.
         model.compile(
            optimizer=Adam(learning_rate=learning_rate_finetune),
            loss='categorical_crossentropy', # Gán Loss Tường minh
            metrics=['accuracy'] # Gán Metrics Tường minh
         )
         return model

    # 2. Bắt đầu Mở khóa (Unfreeze)
    
    # Mở khóa 50 lớp cuối của Xception
    for layer in xception_base.layers[-50:]: 
        if not isinstance(layer, BatchNormalization):
            layer.trainable = True
        
    # Mở khóa 50 lớp cuối của ResNet50
    for layer in resnet_base.layers[-50:]: 
        if not isinstance(layer, BatchNormalization):
            layer.trainable = True

    print(f"--- ĐÃ MỞ KHÓA 50 lớp cuối của Xception và ResNet50. ---")

    # 3. Biên dịch lại (Recompile) với Tốc độ học tập mới và Loss Tường minh
    model.compile(
        optimizer=Adam(learning_rate=learning_rate_finetune),
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )
    
    return model