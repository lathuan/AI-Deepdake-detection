# model_arch.py

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

def create_mesonet(input_width, input_height, input_channels, num_classes):
    """
    Tạo mô hình Mesonet cho phân loại Deepfake.
    """
    model = Sequential()
    
    # Block 1
    model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(input_height, input_width, input_channels), padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    
    # Block 2
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    
    # Block 3
    model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    
    # Block 4
    model.add(Conv2D(128, (5, 5), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    
    # Phần Fully Connected
    model.add(Flatten())
    model.add(Dropout(0.5))
    
    # Output Layer
    # Sử dụng 'softmax' với NUM_CLASSES=2 để đầu ra là 2 xác suất (REAL và FAKE)
    model.add(Dense(num_classes, activation='softmax'))
    
    return model