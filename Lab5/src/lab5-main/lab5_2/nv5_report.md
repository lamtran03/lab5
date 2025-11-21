# NV5 bảng so sánh
## Bảng
| Pipeline                           | F1-score (Macro) | Test Loss  |
| ---------------------------------- | ---------------- | ---------- |
| **TF-IDF + Logistic Regression**   | **0.84**         | N/A        |
| **Word2Vec (Avg) + Dense**         | **0.13**         | **3.1705** |
| **Embedding (Pre-trained) + LSTM** | **0.81**         | **0.7554** |
| **Embedding (Scratch) + LSTM**     | **0.80**         | **0.7450** |

## Nhận xét:

- Mô hình TF-IDF + Logistic Regression đạt F1 cao nhất (0.84), chứng tỏ pipeline cổ điển này vẫn rất mạnh khi dữ liệu nhỏ, câu ngắn, ít phụ thuộc ngữ cảnh.

- Word2Vec + Dense thất bại nặng (F1 = 0.13) → biểu diễn trung bình làm mất hoàn toàn thông tin thứ tự, mô hình không thể học ngữ cảnh.

- LSTM với Embedding (Pretrained) và LSTM học từ đầu (Scratch) đều cho kết quả tốt (~0.81), gần bằng baseline TF-IDF nhưng học được thứ tự chuỗi.

- Dù Test Loss của mô hình học từ đầu nhỏ hơn chút, nhưng F1 thấp hơn → có thể overfit hoặc học embedding chưa đủ tốt.

## Ưu nhược của các mô hình
1️ TF-IDF + Logistic Regression

Ưu điểm:

Huấn luyện nhanh, đơn giản, hiệu năng cao (F1 = 0.84).
Ổn định, ít overfit, mạnh trên dữ liệu nhỏ và câu ngắn.

Nhược điểm:

Không hiểu ngữ cảnh hoặc thứ tự từ.
Dễ nhầm với câu phủ định hoặc cấu trúc phức tạp.

2️ Word2Vec (Average) + Dense Layer

Ưu điểm:

Biểu diễn từ có ngữ nghĩa, mô hình đơn giản, dễ huấn luyện.

Nhược điểm:

Mất toàn bộ thông tin thứ tự, F1 rất thấp (0.13).
Không hiểu phủ định hoặc quan hệ xa giữa các từ.

3️ Embedding (Pre-trained) + LSTM

Ưu điểm:

Hiểu tốt ngữ cảnh, thứ tự từ và phủ định.

Tận dụng embedding học sẵn nên ổn định và chính xác (F1 ≈ 0.81).

Nhược điểm:

Huấn luyện chậm, cần GPU.

Dễ overfit nếu dữ liệu nhỏ, phụ thuộc chất lượng embedding sẵn.

4️ Embedding (Scratch) + LSTM

Ưu điểm:

Học embedding phù hợp với domain, kết quả tốt (F1 ≈ 0.80, loss thấp nhất).

Không phụ thuộc embedding ngoài, linh hoạt khi huấn luyện.

Nhược điểm:

Cần nhiều dữ liệu để học tốt embedding.

Hiệu năng dao động, dễ overfit với tập nhỏ.

# Implementation
## Bước 0
Bước 0 – Chuẩn bị dữ liệu

Giải nén dữ liệu hwu.tar.gz và đọc bằng pandas.
Ba file dữ liệu: hwu_train.csv, hwu_val.csv, hwu_test.csv (không header, phân tách bằng tab).
Dùng LabelEncoder để mã hóa các nhãn intent sang số.
Dữ liệu sau khi xử lý được lưu lại dưới dạng X_train, y_train, X_val, y_val, X_test, y_test.

## NV1
Mục tiêu: xây dựng pipeline truyền thống để có baseline so sánh.
Dùng TfidfVectorizer(max_features=5000) để trích xuất đặc trưng bag-of-words.
Kết hợp với LogisticRegression(max_iter=1000) trong pipeline.
Huấn luyện mô hình trên tập train, đánh giá trên test bằng classification_report.
Lưu pipeline để sử dụng lại trong phần đánh giá định tính.

## NV2 
Dùng gensim.models.Word2Vec để huấn luyện embedding từ dữ liệu huấn luyện.
Viết hàm sentence_to_avg_vector(text, model) để lấy trung bình vector các từ trong câu.
Chuyển toàn bộ dữ liệu train/val/test sang dạng trung bình vector.
Xây dựng mô hình Sequential.
Compile (adam, categorical_crossentropy), huấn luyện và đánh giá F1-score.

## NV3
Dùng Tokenizer của Keras để tạo vocab và chuyển văn bản thành chuỗi chỉ số.
Dùng mô hình Word2Vec từ Nhiệm vụ 2 để tạo ma trận embedding khởi tạo cho LSTM.
Xây dựng mô hình, compile, huấn luyện với EarlyStopping(monitor='val_loss') để tránh overfitting.

## NV4
Sử dụng lại dữ liệu đã tokenized và padded từ nhiệm vụ 3.
Không dùng trọng số từ Word2Vec, để Embedding tự học:
Compile và huấn luyện tương tự nhiệm vụ 3.
Theo dõi val_loss để dừng sớm và lưu mô hình tốt nhất.

# Cách chạy code
Chạy trong src từ 0_ ... đến 4_ ... theo đúng thứ tự

# Khó khăn và giải pháp
NV3 chạy kết quả kém (hầu như toàn 0.00) => Tăng batch size và số epoch , thêm lớp dense và tăng drop out để tránh overfit 
NV4 Chạy kết quả kém : Do chưa lưu và dùng lại kết quả tiền xử lý ở nv3 => Đã lưu lại dữ liệu tiền xử lý ở nv3 sau đó chỉnh lại nv 4
NV5 Chưa rõ cách phân tích định tính 

# References 
https://www.geeksforgeeks.org/machine-learning/understanding-tf-idf-term-frequency-inverse-document-frequency/
https://www.geeksforgeeks.org/machine-learning/text-classification-using-logistic-regression/
https://www.youtube.com/watch?v=hQwFeIupNP0 (Explain word2vec + practical example)

