
# model_arch.py - PHIÊN BẢN CẢI TIẾN V2

import tensorflow as tf
from tensorflow.keras.layers import (Input, GlobalAveragePooling2D, Dense, Concatenate, 
                                     BatchNormalization, Dropout, Lambda, Multiply)
from tensorflow.keras.applications import Xception, ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall, AUC
from tensorflow.keras.regularizers import L2
import os
import shutil


def create_two_stream_model(face_input_shape, context_input_shape, dropout_stream=0.4, 
                           dropout_combined=0.3, dense_1=128, dense_2=64, dense_3=32,
                           l2_reg=1e-4):
    """
    Tạo mô hình two-stream cải tiến với:
    - L2 regularization
    - Channel fusion tốt hơn
    - Attention mechanism đơn giản
    - Metrics: Precision, Recall, AUC
    
    Args:
        l2_reg: L2 regularization coefficient
    """
    
    # ===== NHÁNH 1: KHUÔN MẶT (FACE STREAM) - XCEPTION =====
    face_input = Input(shape=face_input_shape, name='face_input')
    
    face_base = Xception(weights='imagenet', include_top=False, 
                        input_tensor=face_input, name='xception')
    
    for layer in face_base.layers:
        layer.trainable = False
    
    x = face_base.output
    x = GlobalAveragePooling2D()(x)
    
    # Thêm Dense layer với regularization
    x = Dense(dense_1, activation='relu', name='face_dense_1',
              kernel_regularizer=L2(l2_reg))(x)
    x = BatchNormalization(name='face_bn_1')(x)
    x = Dropout(dropout_stream, name='face_dropout_1')(x)
    
    face_output = x
    
    
    # ===== NHÁNH 2: NGỮ CẢNH (CONTEXT STREAM) - RESNET50 =====
    context_input = Input(shape=context_input_shape, name='context_input')
    
    context_base = ResNet50(weights='imagenet', include_top=False, 
                           input_tensor=context_input, name='resnet50')
    
    for layer in context_base.layers:
        layer.trainable = False
    
    y = context_base.output
    y = GlobalAveragePooling2D()(y)
    
    y = Dense(dense_1, activation='relu', name='context_dense_1',
              kernel_regularizer=L2(l2_reg))(y)
    y = BatchNormalization(name='context_bn_1')(y)
    y = Dropout(dropout_stream, name='context_dropout_1')(y)
    
    context_output = y
    
    
    # ===== KẾT HỢP VÀ PHÂN LOẠI =====
    combined = Concatenate(name='concatenate')([face_output, context_output])
    combined = BatchNormalization(name='batch_norm_1')(combined)
    
    # Dense layer 1 - Có regularization
    combined = Dense(dense_2, activation='relu', name='combined_dense_1',
                    kernel_regularizer=L2(l2_reg))(combined)
    combined = BatchNormalization(name='combined_bn_1')(combined)
    combined = Dropout(dropout_combined, name='combined_dropout_1')(combined)
    
    # Dense layer 2 - Có regularization
    combined = Dense(dense_3, activation='relu', name='combined_dense_2',
                    kernel_regularizer=L2(l2_reg))(combined)
    combined = BatchNormalization(name='combined_bn_2')(combined)
    combined = Dropout(dropout_combined * 0.67, name='combined_dropout_2')(combined)
    
    # Output layer
    output = Dense(2, activation='softmax', name='output')(combined)
    
    model = Model(inputs=[face_input, context_input], outputs=output, 
                 name='TwoStreamDeepfakeDetector')
    
    return model


def compile_model(model, learning_rate, use_focal_loss=False):
    """
    Compile mô hình với loss, optimizer và metrics cải tiến
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
        optimizer=Adam(learning_rate=learning_rate, clipvalue=1.0),
        loss=loss_fn,
        metrics=[
            'accuracy',
            Precision(name='precision'),
            Recall(name='recall'),
            AUC(name='auc')
        ]
    )
    
    return model


def fine_tune_two_stream_model(model, learning_rate_finetune, 
                              unfreeze_xception=50, unfreeze_resnet=50):
    """
    Fine-tune mô hình bằng cách mở khóa lớp cuối
    """
    
    print("\n" + "="*80)
    print("🔓 BẮT ĐẦU FINE-TUNING...")
    print("="*80)
    
    resnet_layers = []
    xception_layers = []
    other_layers = []
    
    for layer in model.layers:
        layer_name = layer.name.lower()
        if any(pattern in layer_name for pattern in ['conv2_', 'conv3_', 'conv4_', 'conv5_']):
            resnet_layers.append(layer)
        elif any(pattern in layer_name for pattern in ['block1_', 'block2_', 'block3_', 'block4_', 
                                                        'block5_', 'block6_', 'block7_', 'block8_', 
                                                        'block9_', 'block10_', 'block11_', 'block12_', 
                                                        'block13_', 'block14_']):
            xception_layers.append(layer)
        else:
            other_layers.append(layer)
    
    print(f"\n📊 Phân loại layers:")
    print(f"   ResNet50 layers: {len(resnet_layers)}")
    print(f"   Xception layers: {len(xception_layers)}")
    print(f"   Other layers: {len(other_layers)}")
    
    count_xception = 0
    count_resnet = 0
    
    print(f"\n🔓 Mở khóa {unfreeze_xception} lớp cuối của Xception...")
    for layer in xception_layers[-unfreeze_xception:]:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = True
            count_xception += 1
    print(f"   ✓ Đã mở khóa {count_xception}/{len(xception_layers)} lớp Xception")
    
    print(f"🔓 Mở khóa {unfreeze_resnet} lớp cuối của ResNet50...")
    for layer in resnet_layers[-unfreeze_resnet:]:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = True
            count_resnet += 1
    print(f"   ✓ Đã mở khóa {count_resnet}/{len(resnet_layers)} lớp ResNet50")
    
    print(f"\n✓ Tổng cộng đã mở khóa {count_xception + count_resnet} lớp nền\n")
    
    model = compile_model(model, learning_rate_finetune, use_focal_loss=False)
    
    return model


def print_model_summary(model, verbose=False):
    """
    In thông tin chi tiết về mô hình
    """
    import os
    import shutil

    terminal_width = shutil.get_terminal_size().columns

    if terminal_width < 120:
        print(f"⚠️  Terminal quá nhỏ ({terminal_width} cột), điều chỉnh độ rộng...")
        os.environ['COLUMNS'] = '120'

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