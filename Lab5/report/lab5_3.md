1. Giải thích code
Task 1
    Viết hàm load_conllu() để đọc file theo định dạng CoNLL-U và trích xuất các cặp (word, tag) theo từng câu.
    Đọc dữ liệu train và dev từ file.
    Xây hai bảng mã hóa:
        word_to_ix: ánh xạ từ → chỉ số, có thêm <UNK> cho từ ngoài từ điển.
        tag_to_ix: ánh xạ nhãn POS → chỉ số.

Task 2
    Tạo lớp POSDataset kế thừa torch.utils.data.Dataset:
    Mỗi mẫu là một câu, được chuyển thành list chỉ số từ và nhãn.
    Dùng pad_sequence trong collate_fn để đệm các câu trong batch về cùng độ dài.
    Tạo DataLoader cho train và dev để mô hình đọc dữ liệu theo batch.

Task 3
    Định nghĩa mô hình SimpleRNNForTokenClassification gồm:
        Embedding: chuyển chỉ số từ thành vector.
        RNN: xử lý chuỗi embedding theo thời gian.
        Linear: dự đoán nhãn POS cho mỗi token.
    Chạy thử model trên một batch và kiểm tra kích thước đầu vào và đầu ra.

Task 4
    Khởi tạo mô hình, optimizer (Adam) và hàm loss (CrossEntropyLoss), bỏ qua nhãn pad bằng ignore_index=0.
    Vòng lặp huấn luyện:
    Forward
    Tính loss
    Backpropagation
    Cập nhật trọng số
    Sau mỗi epoch in loss trung bình.
    In thử một câu trong dev và dự đoán nhãn.

Task 5
    viết hàm đo accuracy token-level trên tập dev 
    Đánh giá sau mỗi epoch 
    Lưu mô hình tốt nhất và in kết quả dự đoán mẫu để kiểm tra chất lượng mô hình sau huấn luyện.

2. Hướng dẫn chạy 
- Chạy file ipynb

3. Đánh giá kết quả

BÁO CÁO KẾT QUẢ:

• Vocabulary size: 20201
• Tag set size: 18

Mô hình RNN được huấn luyện 3 epoch.
Sau mỗi epoch,hàm loss, độ chính xác trên tập train và dev như sau:

Epoch 1: Loss=0.9911 | Train Acc=0.7048 | Dev Acc=0.8074
Epoch 2: Loss=0.5086 | Train Acc=0.8412 | Dev Acc=0.8508
Epoch 3: Loss=0.3749 | Train Acc=0.8818 | Dev Acc=0.8724

Dựa trên độ chính xác trên tập dev, mô hình có điểm dev cao nhất được lưu lại.
Sau khi load mô hình tốt nhất:

Final Dev Accuracy: 0.8724130663856692

Hàm predict_sentence(sentence) được viết để dự đoán nhãn UPOS cho câu mới.

KẾT QUẢ THỰC HIỆN :
- Độ chính xác trên tập dev : 0.8724...
- Ví dụ dự đoán câu mới :
  + Câu "i love NLP"
  + Dự đoán : [('i', 'PRON'), ('love', 'VERB'), ('NLP', 'VERB')]
  + Câu "This is a test sentence"
  + Dự đoán : [('This', 'PRON'), ('is', 'AUX'), ('a', 'DET'), ('test', 'NOUN'), ('sentence', 'VERB')]

ĐÁNH GIÁ KẾT QUẢ
    Mô hình RNN được huấn luyện trong 3 epoch và kết quả cho thấy:
    Loss giảm đều qua từng epoch, chứng tỏ mô hình học ổn định.
    Train accuracy tăng từ 70% → 88%, thể hiện mô hình ghi nhớ tốt hơn dữ liệu huấn luyện.
    Dev accuracy tăng từ 80% → 87%, cho thấy mô hình tổng quát hóa tốt trên dữ liệu chưa thấy.
    Chênh lệch train/dev không quá lớn → không có dấu hiệu overfitting rõ rệt.
    Mô hình đạt Final Dev Accuracy ≈ 87.2%, khá tốt với RNN đơn giản và không dùng thêm các kỹ thuật nâng cao (như BiLSTM, CRF hoặc pretrained embeddings).
    Việc lưu mô hình có dev accuracy cao nhất giúp tránh việc lưu mô hình kém nhất do biến động trong huấn luyện.

4. Khó khăn và giải pháp
5. Trích dẫn tài liệu
https://docs.pytorch.org/tutorials/beginner/nlp/
tài liệu lab5_3
Chatgpt