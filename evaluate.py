# Code đánh giá mô hình
# evaluate.py - ĐÁNH GIÁ & TEST MÔ HÌNH

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                            roc_curve, auc, precision_recall_curve, f1_score)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from config import *


# --- HÀM TẠO TEST GENERATOR ---
def get_test_generator(data_dir, target_size_face, target_size_context, batch_size, 
                      validation_split=VALIDATION_SPLIT):
    """
    Tạo generator cho dữ liệu test (validation set)
    """
    datagen = ImageDataGenerator(rescale=1./255, validation_split=validation_split)
    
    face_gen = datagen.flow_from_directory(
        data_dir,
        target_size=target_size_face,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        seed=42
    )
    
    context_gen = datagen.flow_from_directory(
        data_dir,
        target_size=target_size_context,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        seed=42
    )
    
    total_samples = face_gen.n
    
    def test_generator():
        while True:
            X_face = face_gen.__next__()
            X_context = context_gen.__next__()
            yield ({'face_input': X_face[0], 'context_input': X_context[0]}, X_face[1])
    
    return test_generator(), total_samples, face_gen.class_indices


# --- HÀM PREDICT VÀ LẤY TRUE LABELS ---
def get_predictions_and_labels(model, test_gen, test_steps, total_samples):
    """
    Lấy predictions từ mô hình và true labels
    
    Returns:
        predictions: Array shape (n_samples, 2) - confidence cho mỗi class
        true_labels: Array shape (n_samples,) - true labels
    """
    predictions = model.predict(test_gen, steps=test_steps, verbose=1)
    
    # Reset generator để lấy labels
    datagen = ImageDataGenerator(rescale=1./255, validation_split=VALIDATION_SPLIT)
    face_gen_for_labels = datagen.flow_from_directory(
        DATA_DIR,
        target_size=(FACE_IMG_WIDTH, FACE_IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        seed=42
    )
    
    true_labels = []
    samples_collected = 0
    while samples_collected < total_samples:
        _, labels_batch = face_gen_for_labels.__next__()
        true_labels.extend(np.argmax(labels_batch, axis=1))
        samples_collected += len(labels_batch)
    
    true_labels = np.array(true_labels[:total_samples])
    predictions = predictions[:total_samples]
    
    return predictions, true_labels


# --- HÀM TÍNH CÁC METRICS ---
def calculate_metrics(predictions, true_labels):
    """
    Tính toàn bộ metrics cho binary classification
    
    Returns:
        Dict chứa các metrics
    """
    # Lấy predicted labels (class có confidence cao nhất)
    pred_labels = np.argmax(predictions, axis=1)
    
    # Confidence cho class Deepfake (class 1)
    pred_probabilities = predictions[:, 1]
    
    # ROC-AUC
    roc_auc = roc_auc_score(true_labels, pred_probabilities)
    
    # F1-Score
    f1 = f1_score(true_labels, pred_labels, average='weighted')
    
    # Accuracy
    accuracy = np.mean(pred_labels == true_labels)
    
    metrics = {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'f1_score': f1,
        'pred_labels': pred_labels,
        'pred_probabilities': pred_probabilities
    }
    
    return metrics


# --- HÀM PRINT CLASSIFICATION REPORT ---
def print_classification_report(predictions, true_labels):
    """
    In chi tiết classification report
    """
    pred_labels = np.argmax(predictions, axis=1)
    
    print("\n" + "="*80)
    print("📊 CLASSIFICATION REPORT")
    print("="*80)
    print(classification_report(true_labels, pred_labels, 
                               target_names=['Real', 'Deepfake'],
                               digits=4))


# --- HÀM PLOT CONFUSION MATRIX ---
def plot_confusion_matrix(predictions, true_labels, save_path=None):
    """
    Vẽ confusion matrix
    """
    pred_labels = np.argmax(predictions, axis=1)
    cm = confusion_matrix(true_labels, pred_labels)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Real', 'Deepfake'],
                yticklabels=['Real', 'Deepfake'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"✓ Lưu confusion matrix: {save_path}")
    plt.show()
    
    return cm


# --- HÀM PLOT ROC CURVE ---
def plot_roc_curve(predictions, true_labels, save_path=None):
    """
    Vẽ ROC curve
    """
    pred_probabilities = predictions[:, 1]
    
    fpr, tpr, _ = roc_curve(true_labels, pred_probabilities)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"✓ Lưu ROC curve: {save_path}")
    plt.show()
    
    return roc_auc


# --- HÀM PLOT PRECISION-RECALL CURVE ---
def plot_precision_recall_curve(predictions, true_labels, save_path=None):
    """
    Vẽ Precision-Recall curve
    """
    pred_probabilities = predictions[:, 1]
    
    precision, recall, _ = precision_recall_curve(true_labels, pred_probabilities)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="upper right")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"✓ Lưu PR curve: {save_path}")
    plt.show()
    
    return pr_auc


# --- HÀM CHÍNH ĐỂ EVALUATE ---
def evaluate_model(model_path, plot_results=True, save_figures=True):
    """
    Đánh giá mô hình trên validation set
    
    Args:
        model_path: Đường dẫn đến model file
        plot_results: Vẽ các biểu đồ
        save_figures: Lưu các biểu đồ
    """
    
    print("\n" + "="*80)
    print("🧪 ĐÁNH GIÁ MÔ HÌNH DEEPFAKE DETECTION")
    print("="*80)
    
    # Load model
    print(f"\n📥 Load mô hình: {model_path}")
    if not os.path.exists(model_path):
        print(f"❌ LỖI: Model không tìm thấy: {model_path}")
        return
    
    model = tf.keras.models.load_model(model_path)
    print("✓ Model loaded successfully")
    
    # Chuẩn bị test generator
    print("\n📂 Chuẩn bị test data...")
    test_gen, total_samples, class_indices = get_test_generator(
        DATA_DIR,
        (FACE_IMG_WIDTH, FACE_IMG_HEIGHT),
        (CONTEXT_IMG_WIDTH, CONTEXT_IMG_HEIGHT),
        BATCH_SIZE
    )
    test_steps = total_samples // BATCH_SIZE
    print(f"✓ Test samples: {total_samples} ({test_steps} steps)")
    print(f"✓ Class indices: {class_indices}")
    
    # Lấy predictions
    print("\n🔮 Lấy predictions...")
    predictions, true_labels = get_predictions_and_labels(model, test_gen, test_steps, total_samples)
    print(f"✓ Predictions shape: {predictions.shape}")
    print(f"✓ True labels shape: {true_labels.shape}")
    
    # Tính metrics
    print("\n📈 Tính toán metrics...")
    metrics = calculate_metrics(predictions, true_labels)
    
    print("\n" + "="*80)
    print("📊 METRICS TỔNG HỢP")
    print("="*80)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    
    # In classification report
    print_classification_report(predictions, true_labels)
    
    # Vẽ biểu đồ
    if plot_results:
        print("\n🎨 Vẽ biểu đồ...")
        
        figures_dir = os.path.join(MODEL_OUTPUT_DIR, 'figures')
        if save_figures and not os.path.exists(figures_dir):
            os.makedirs(figures_dir)
        
        # Confusion Matrix
        cm_path = os.path.join(figures_dir, 'confusion_matrix.png') if save_figures else None
        plot_confusion_matrix(predictions, true_labels, save_path=cm_path)
        
        # ROC Curve
        roc_path = os.path.join(figures_dir, 'roc_curve.png') if save_figures else None
        plot_roc_curve(predictions, true_labels, save_path=roc_path)
        
        # Precision-Recall Curve
        pr_path = os.path.join(figures_dir, 'pr_curve.png') if save_figures else None
        plot_precision_recall_curve(predictions, true_labels, save_path=pr_path)
    
    print("\n" + "="*80)
    print("✓ HOÀN THÀNH ĐÁNH GIÁ")
    print("="*80 + "\n")
    
    return metrics


# --- DEMO PREDICTION TRÊN 1 ẢNH ---
def predict_single_image(model_path, face_image_path, context_image_path):
    """
    Predict deepfake trên 1 cặp ảnh (face + context)
    
    Args:
        model_path: Đường dẫn model
        face_image_path: Đường dẫn ảnh face
        context_image_path: Đường dẫn ảnh context
    
    Returns:
        Predicted class (0=Real, 1=Deepfake) và confidence
    """
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    
    model = tf.keras.models.load_model(model_path)
    
    # Load ảnh
    face_img = load_img(face_image_path, target_size=(FACE_IMG_WIDTH, FACE_IMG_HEIGHT))
    context_img = load_img(context_image_path, target_size=(CONTEXT_IMG_WIDTH, CONTEXT_IMG_HEIGHT))
    
    # Preprocess
    face_arr = img_to_array(face_img) / 255.0
    context_arr = img_to_array(context_img) / 255.0
    
    # Predict
    face_batch = np.expand_dims(face_arr, axis=0)
    context_batch = np.expand_dims(context_arr, axis=0)
    
    prediction = model.predict({'face_input': face_batch, 'context_input': context_batch})
    
    class_label = ['Real', 'Deepfake']
    pred_class = np.argmax(prediction[0])
    confidence = prediction[0][pred_class]
    
    print(f"\n🔮 Prediction:")
    print(f"   Class: {class_label[pred_class]}")
    print(f"   Confidence: {confidence:.4f}")
    print(f"   Probabilities: Real={prediction[0][0]:.4f}, Deepfake={prediction[0][1]:.4f}")
    
    return pred_class, confidence


if __name__ == '__main__':
    # Đánh giá mô hình
    model_path = os.path.join(MODEL_OUTPUT_DIR, MODEL_NAME)
    
    metrics = evaluate_model(model_path, plot_results=True, save_figures=True)