# model_arch.py

import tensorflow as tf
from tensorflow.keras.applications import Xception, ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input, concatenate
from tensorflow.keras.optimizers import Adam

# --- Định nghĩa Hàm Tạo Mô hình (create_two_stream_model) ---
def create_two_stream_model(face_input_shape, context_input_shape):
    
    # NHÁNH 1: KHUÔN MẶT (Xception)
    face_input = Input(shape=face_input_shape, name='face_input')
    face_base = Xception(
        weights='imagenet', 
        include_top=False, 
        input_tensor=face_input,
        input_shape=face_input_shape,
        # Đặt tên lớp cho nhánh này để dễ dàng truy cập sau
        name='xception_base' 
    )
    for layer in face_base.layers:
        layer.trainable = False
        
    face_output = face_base.output
    face_output = GlobalAveragePooling2D(name='face_avg_pool')(face_output)
    
    # NHÁNH 2: NGỮ CẢNH (ResNet50)
    context_input = Input(shape=context_input_shape, name='context_input')
    context_base = ResNet50(
        weights='imagenet',
        include_top=False,
        input_tensor=context_input,
        input_shape=context_input_shape,
        # Đặt tên lớp cho nhánh này
        name='resnet50_base'
    )
    for layer in context_base.layers:
        layer.trainable = False
        
    context_output = context_base.output
    context_output = GlobalAveragePooling2D(name='context_avg_pool')(context_output)
    
    # HỢP NHẤT (FUSION) và PHÂN LOẠI
    merged = concatenate([face_output, context_output], name='fusion_layer')
    merged = Dense(512, activation='relu', name='dense_fusion')(merged)
    merged = Dropout(0.5, name='dropout_fusion')(merged)
    predictions = Dense(2, activation='softmax', name='output_layer')(merged)

    model = Model(inputs=[face_input, context_input], outputs=predictions)
    
    model.compile(
        optimizer=Adam(learning_rate=0.0001), 
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return model


# --- Định nghĩa Fine-Tuning (fine_tune_two_stream_model) ---
def fine_tune_two_stream_model(model, unfreeze_xception_from_layer=-50, unfreeze_resnet_from_layer=-20):
    
    # Lấy các lớp nền dựa trên tên đã đặt
    try:
        face_base = model.get_layer('xception_base') 
        context_base = model.get_layer('resnet50_base') 
    except ValueError:
        print("LỖI: Không thể tìm thấy lớp nền Xception/ResNet. Kiểm tra lại tên lớp.")
        return model

    # Mở khóa các lớp cuối của Xception
    for layer in face_base.layers[unfreeze_xception_from_layer:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True

    # Mở khóa các lớp cuối của ResNet
    for layer in context_base.layers[unfreeze_resnet_from_layer:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True

    # Biên dịch lại mô hình với Learning Rate CỰC KỲ NHỎ
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), 
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    print("\n--- Bắt đầu Fine-Tuning Two-Stream Model ---")
    return model