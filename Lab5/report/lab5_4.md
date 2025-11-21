1. Giải thích code 
Task 1 – Tải và xử lý dữ liệu

Dùng load_dataset() để tải CoNLL-2003.
Lấy danh sách:
tokens cho mỗi câu.
ner_tags cho mỗi câu.
Do dataset không có label mapping sẵn, ta tự định nghĩa mapping chuẩn CoNLL-2003.
Xây dựng:
word_to_ix: từ → chỉ số.
tag_to_ix: nhãn NER → chỉ số.
Thêm 2 token đặc biệt:
<PAD> để padding.
<UNK> cho từ lạ.
Mục tiêu: chuyển dữ liệu text về dạng số để đưa vào mô hình.

Task 2 – Dataset & DataLoader
Tạo lớp NERDataset kế thừa Dataset:
__getitem__ trả về:
list chỉ số từ
list chỉ số nhãn
Viết collate_fn:
pad các câu trong batch về cùng độ dài
pad nhãn bằng -1 (TAG_PAD)
dùng pad_sequence(batch_first=True)
Tạo DataLoader cho train và validation.
Mục tiêu: tạo pipeline để dữ liệu đưa vào mô hình theo batch.

Task 3 – Xây mô hình RNN
Model gồm:
nn.Embedding
chuyển từ chỉ số → vector embedding
nn.LSTM (hoặc RNN/GRU)
xử lý chuỗi
nn.linear
đưa output LSTM ra số lớp NER
Forward:
Nhận batch [batch_size, seq_len]
Trả ra [batch_size, seq_len, num_tags]
Mục tiêu: xây mô hình sequence labeling.

Task 4 – Huấn luyện mô hình
Dùng:
Adam optimizer
CrossEntropyLoss(ignore_index=TAG_PAD)
Mỗi epoch:
Zero grad
Forward
Tính loss
Backward
Update weight
In loss trung bình mỗi epoch.
Mục tiêu: tối ưu trọng số mô hình qua nhiều vòng lặp.

Task 5 – Đánh giá & dự đoán
Hàm evaluate()
model.eval(), tắt gradient.
Dự đoán bằng argmax.
Tính accuracy nhưng chỉ với token không phải padding.
Hàm predict_sentence()
Tách câu thành token.
Chuyển sang chỉ số.
Model dự đoán.
In ra từng cặp (từ, nhãn_dự_đoán).
Mục tiêu:
Kiểm tra độ chính xác trên validation.

Cho phép dự đoán NER trên câu mới.
2. Hướng dẫn chạy code
- Chạy file ipynb

3. Khó khăn và giải pháp
- Dataset tài liệu đưa ra không được hỗ trợ trong huggingface nữa => Tìm data từ maintainer khác : lhoestq/conll2003.
- Các phiên bản CoNLL-2003 khác nhau trên HuggingFace có cách lưu label khác nhau (có bản lưu ClassLabel, có bản chỉ lưu Value). Điều này gây lỗi khi cố truy .names. => Tự định nghĩa bảng nhãn (như 9 nhãn chuẩn CoNLL-2003).
- Xây dựng vocabulary dễ bị thiếu hoặc sai index => Luôn thêm <PAD> và <UNK> vào word_to_ix, khi chuyển token → id: dùng .get(word, UNK_IDX).

4. Tài liệu tham khảo
https://huggingface.co/datasets/eriktks/conll2003
Tài liệu lab5
Chatgpt