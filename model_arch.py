# model_arch.py - PHIÊN BẢN CẢI THIỆN (THÊM DROPOUT & LỚPMỚI)

import tensorflow as tf
from tensorflow.keras.layers import (Input, GlobalAveragePooling2D, Dense, 
                                     Concatenate, BatchNormalization, Dropout)
from tensorflow.keras.applications import Xception, ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall, AUC


# --- HÀM TẠO MÔ HÌNH TWO-STREAM (CẢI THIỆN) ---
def create_two_stream_model(face_input_shape, context_input_shape, dropout_stream=0.4, 
                           dropout_combined=0.3, dense_1=128, dense_2=64, dense_3=32):
    """
    Tạo mô hình two-stream với:
    - Dropout để tránh overfitting
    - Lớp Dense trung gian thêm
    - Metrics: Precision, Recall, AUC
    
    Args:
        face_input_shape: Tuple (height, width, channels) cho Face stream
        context_input_shape: Tuple (height, width, channels) cho Context stream
        dropout_stream: Dropout rate sau Face/Context output
        dropout_combined: Dropout rate sau combined layers
        dense_1: Số unit cho dense layer 1 (Face & Context output)
        dense_2: Số unit cho dense layer 2 (combined)
        dense_3: Số unit cho dense layer 3 (combined)
    
    Returns:
        Keras Model
    """
    
    # ===== NHÁNH 1: KHUÔN MẶT (FACE STREAM) - XCEPTION =====
    face_input = Input(shape=face_input_shape, name='face_input')
    
    # Tải Xception pretrained
    face_base = Xception(weights='imagenet', include_top=False, 
                        input_tensor=face_input, name='xception')
    
    # Khóa toàn bộ lớp ban đầu
    for layer in face_base.layers:
        layer.trainable = False
    
    # Xử lý output
    x = face_base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(dense_1, activation='relu', name='face_dense_1')(x)
    x = Dropout(dropout_stream, name='face_dropout_1')(x)  # THÊM DROPOUT
    face_output = x
    
    
    # ===== NHÁNH 2: NGỮ CẢNH (CONTEXT STREAM) - RESNET50 =====
    context_input = Input(shape=context_input_shape, name='context_input')
    
    # Tải ResNet50 pretrained
    context_base = ResNet50(weights='imagenet', include_top=False, 
                           input_tensor=context_input, name='resnet50')
    
    # Khóa toàn bộ lớp ban đầu
    for layer in context_base.layers:
        layer.trainable = False
    
    # Xử lý output
    y = context_base.output
    y = GlobalAveragePooling2D()(y)
    y = Dense(dense_1, activation='relu', name='context_dense_1')(y)
    y = Dropout(dropout_stream, name='context_dropout_1')(y)  # THÊM DROPOUT
    context_output = y
    
    
    # ===== KẾT HỢP VÀ PHÂN LOẠI =====
    combined = Concatenate(name='concatenate')([face_output, context_output])
    combined = BatchNormalization(name='batch_norm_1')(combined)
    
    # Dense layer 1
    combined = Dense(dense_2, activation='relu', name='combined_dense_1')(combined)
    combined = Dropout(dropout_combined, name='combined_dropout_1')(combined)  # THÊM DROPOUT
    
    # Dense layer 2 (LỚP MỚI THÊM VÀO)
    combined = Dense(dense_3, activation='relu', name='combined_dense_2')(combined)
    combined = Dropout(dropout_combined * 0.67, name='combined_dropout_2')(combined)  # DROPOUT CHA HƠN
    
    # Output layer
    output = Dense(2, activation='softmax', name='output')(combined)
    
    # Tạo model
    model = Model(inputs=[face_input, context_input], outputs=output, 
                 name='TwoStreamDeepfakeDetector')
    
    return model


# --- HÀM COMPILE MÔ HÌNH VỚI METRICS CẢI THIỆN ---
def compile_model(model, learning_rate, use_focal_loss=False):
    """
    Compile mô hình với loss, optimizer và metrics cải thiện
    
    Args:
        model: Keras model
        learning_rate: Learning rate cho optimizer
        use_focal_loss: Sử dụng Focal Loss (tốt hơn cho imbalanced data)
                       Nếu False, dùng categorical crossentropy
    
    Returns:
        Compiled model
    """
    
    if use_focal_loss:
        try:
            import tensorflow_addons as tfa
            loss_fn = tfa.losses.SigmoidFocalCrossEntropy()
            print("✓ Dùng Focal Loss cho imbalanced data")
        except ImportError:
            print("⚠ tensorflow_addons không được cài. Sử dụng Categorical Crossentropy")
            loss_fn = tf.keras.losses.CategoricalCrossentropy()
    else:
        loss_fn = tf.keras.losses.CategoricalCrossentropy()
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=loss_fn,
        metrics=[
            'accuracy',
            Precision(name='precision'),  # THÊM PRECISION
            Recall(name='recall'),         # THÊM RECALL
            AUC(name='auc')               # THÊM AUC
        ]
    )
    
    return model


# --- HÀM TINH CHỈNH (FINE-TUNING) - CẢI THIỆN =====
def fine_tune_two_stream_model(model, learning_rate_finetune, 
                              unfreeze_xception=50, unfreeze_resnet=50):
    """
    Fine-tune mô hình bằng cách mở khóa lớp cuối
    
    Args:
        model: Trained model
        learning_rate_finetune: Learning rate cho fine-tuning
        unfreeze_xception: Số lớp cuối của Xception để mở khóa
        unfreeze_resnet: Số lớp cuối của ResNet50 để mở khóa
    
    Returns:
        Model sau khi fine-tune
    """
    
    xception_base = None
    resnet_base = None
    
    # CÓ CHỂ 1: Tìm bằng isinstance + name
    for layer in model.layers:
        if isinstance(layer, Xception) and layer.name == 'xception':
            xception_base = layer
        elif isinstance(layer, ResNet50) and layer.name == 'resnet50':
            resnet_base = layer
    
    # CƠ CHẾ 2: Tìm bằng get_layer (nếu cơ chế 1 thất bại)
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
    
    # LỖI: Không tìm thấy base models
    if xception_base is None or resnet_base is None:
        print("❌ LỖI: KHÔNG THỂ TINH CHỈNH. Thất bại khi tìm lớp nền Xception/ResNet50.")
        print("⚠ Sẽ dùng mô hình hiện tại mà không fine-tune")
        
        # Compile lại với loss & metrics tường minh
        model = compile_model(model, learning_rate_finetune)
        return model
    
    # MỞ KHÓA CÁC LỚP CUỐI
    print(f"🔓 Mở khóa {unfreeze_xception} lớp cuối của Xception...")
    for layer in xception_base.layers[-unfreeze_xception:]:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = True
    
    print(f"🔓 Mở khóa {unfreeze_resnet} lớp cuối của ResNet50...")
    for layer in resnet_base.layers[-unfreeze_resnet:]:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = True
    
    print(f"✓ Đã mở khóa {unfreeze_xception + unfreeze_resnet} lớp nền")
    
    # COMPILE LẠI VỚI LEARNING RATE MỚI
    model = compile_model(model, learning_rate_finetune, use_focal_loss=False)
    
    return model


# --- HÀM IN CẤU TRÚC MÔ HÌNH (DEBUG) ---
def print_model_summary(model, verbose=False):
    """
    In thông tin chi tiết về mô hình
    """
    print("\n" + "="*80)
    print("📊 THÔNG TIN MÔ HÌNH")
    print("="*80)
    
    model.summary()
    
    if verbose:
        print("\n📋 CHI TIẾT CÁC LỚP:")
        for i, layer in enumerate(model.layers):
            trainable = "🔓" if layer.trainable else "🔒"
            params = layer.count_params()
            print(f"  {i:2d}. {trainable} {layer.name:30s} | {layer.__class__.__name__:20s} | {params:>12,} params")
    
    print("="*80 + "\n")