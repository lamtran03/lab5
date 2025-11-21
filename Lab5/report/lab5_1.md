1. Giải thích code
Task 1
- Tạo tensor từ list Python và từ NumPy array.
- Tạo tensor mới có cùng kích thước nhưng chứa toàn số 1 hoặc giá trị ngẫu nhiên.
- Thực hiện các thao tác toán học cơ bản:
- Cộng tensor
- Nhân vô hướng
- Nhân ma trận
- Truy cập phần tử theo chỉ số (lấy hàng, cột, hoặc phần tử cụ thể).
- Thay đổi hình dạng tensor bằng view.

Task 2
- Tạo một tensor có requires_grad=True để PyTorch theo dõi các phép tính.
- Tạo biểu thức toán học dựa trên tensor đó.
- Gọi .backward() để PyTorch tự động tính đạo hàm.
- Lấy gradient bằng x.grad.

Task 3
- Linear layer: nhập tensor và biến đổi tuyến tính để cho ra output có số chiều xác định.
- Embedding layer: ánh xạ các chỉ số từ (token ID) thành vector embedding có số chiều nhỏ hơn và giàu thông tin hơn.
- Mô hình nhỏ (MyFirstModel):
    - Embedding → Linear → ReLU → Linear
    - Nhận câu gồm các token (int), chuyển thành embedding, tính hidden và dự đoán đầu ra.

2. Hướng dẫn chạy code
- Chạy file ipynb

3. Đánh giá kết quả
- Reshape từ 4x4 sang 16x1 đúng và giữ nguyên dữ liệu
- Kết quả đạo hàm đúng
- Output của mô hình có shape [1, 4, 2], phù hợp:
    - Batch = 1
    - Sequence length = 4
    - Output dimension = 2

4. Khó khăn và giải pháp
- Tokenization mismatch giữa training và inference (ví dụ split khác) => đảm bảo cùng tokenizer/chuẩn split khi huấn luyện và predict.
- Long sequences / memory => giới hạn max_seq_len hoặc dùng packed sequences (pack_padded_sequence) để tiết kiệm.

5. Trích dẫn tài liệu 
https://docs.pytorch.org/docs/stable/nn.html
tài liệu lab5.1
Chatgpt