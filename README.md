# Dự án Deepfake Detection
📋 I. TỔNG QUAN
Tên dự án:
Website for Deepfake Video Detection
Mục tiêu:
Xây dựng hệ thống AI phát hiện video deepfake sử dụng mô hình deep learning hai nhánh, kết hợp phân tích khuôn mặt và bối cảnh để đạt độ chính xác cao.
Ứng dụng:

Kiểm soát thông tin giả mạo trên mạng xã hội
Bảo vệ danh tính và an toàn cá nhân
Hỗ trợ điều tra pháp tố
Kiểm chứng xác thực nội dung video


🎯 II. BÀI TOÁN
Deepfake là những video giả mạo được tạo bằng kỹ thuật AI, đặc biệt là GAN (Generative Adversarial Networks). Mặc dù công nghệ deepfake ngày càng tinh vi, nhưng vẫn để lại những dấu hiệu nhỏ trên khuôn mặt và bối cảnh. Dự án này nhằm phát hiện những dấu hiệu đó để xác định video là thật hay giả mạo.

🏗️ III. KIẾN TRÚC SYSTEM
Mô hình Two-Stream:
Mô hình được chia thành 2 nhánh xử lý độc lập:
Nhánh 1: Face Stream

Nhận đầu vào là ảnh khuôn mặt (320×320)
Sử dụng mạng Xception (pretrained trên ImageNet)
Phân tích các đặc trưng liên quan đến khuôn mặt

Nhánh 2: Context Stream

Nhận đầu vào là toàn bộ frame (224×224)
Sử dụng mạng ResNet50 (pretrained trên ImageNet)
Phân tích bối cảnh xung quanh

Sau đó, hai nhánh được kết hợp lại thông qua các lớp Dense để đưa ra quyết định cuối cùng: Real hoặc Deepfake.

📚 IV. PHƯƠNG PHÁP HUẤN LUYỆN
Transfer Learning:
Sử dụng những mô hình đã được huấn luyện trên ImageNet (một dataset khổng lồ với 1 triệu ảnh). Điều này giúp mô hình có thể học các đặc trưng chung (edges, textures, shapes) mà không cần phải train từ đầu.
2-Phase Training:
Phase 1 - Warm-up (15 epochs):

Khóa các lớp nền (frozen), chỉ huấn luyện các lớp trên cùng
Learning rate cao để học nhanh
Kết quả: ~88% accuracy trên validation set

Phase 2 - Fine-tuning (25 epochs):

Mở khóa 50 lớp cuối cùng của mỗi base model
Learning rate thấp để fine-tune nhẹ nhàng
Kết quả: ~92% accuracy trên validation set

Phương pháp này giúp mô hình học nhanh hơn và hiệu quả hơn so với huấn luyện từ đầu.

📊 V. DỮ LIỆU
Dataset được chia thành hai lớp:

Real: Ảnh/video thực từ những người thật
Deepfake: Ảnh/video giả mạo tạo bằng AI

Dữ liệu được chia 80% cho training và 20% cho validation. Áp dụng các kỹ thuật tăng cường dữ liệu (data augmentation) như xoay, dịch, zoom để tăng độ đa dạng.

🧪 VI. KẾT QUẢ
Mô hình đạt được:

Accuracy: 92% trên validation set
Precision & Recall: > 90%
ROC-AUC: 0.96

Mô hình có khả năng phát hiện deepfake trong video thông qua việc cắt các frame quan trọng (3-5 giây) và phân tích chúng.

💡 VII. CÓ PHƯƠNG PHÁP CHÍNH
Các kỹ thuật được áp dụng:

Transfer Learning: Tái sử dụng các mô hình đã được huấn luyện sẵn
Dropout: Ngẫu nhiên tắt một số neurons để tránh overfitting
Batch Normalization: Chuẩn hóa input để ổn định training
Data Augmentation: Tăng cường dữ liệu bằng các biến đổi ảnh
EarlyStopping: Dừng huấn luyện sớm khi không còn cải thiện
ReduceLROnPlateau: Giảm learning rate khi loss không giảm
