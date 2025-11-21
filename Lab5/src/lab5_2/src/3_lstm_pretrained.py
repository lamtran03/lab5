import sys
sys.stdout.reconfigure(encoding='utf-8')


import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import classification_report
import joblib


# 1. Đọc và chuẩn bị dữ liệu

train_path = "lab5_2/data/hwu/train_processed.csv"
val_path   = "lab5_2/data/hwu/val_processed.csv"
test_path  = "lab5_2/data/hwu/test_processed.csv"

df_train = pd.read_csv(train_path)
df_val   = pd.read_csv(val_path)
df_test  = pd.read_csv(test_path)

y_train = df_train["label"].values
y_val   = df_val["label"].values
y_test  = df_test["label"].values
num_classes = len(np.unique(y_train))

# 1a. Tokenizer: Tạo vocab và chuyển text thành chuỗi chỉ số


tokenizer = Tokenizer(oov_token="<UNK>")
tokenizer.fit_on_texts(df_train["text"])

train_sequences = tokenizer.texts_to_sequences(df_train["text"])
val_sequences   = tokenizer.texts_to_sequences(df_val["text"])
test_sequences  = tokenizer.texts_to_sequences(df_test["text"])

# 1b. Padding: Đảm bảo các chuỗi có cùng độ dài
max_len = 60
X_train_pad = pad_sequences(train_sequences, maxlen=max_len, padding='post', truncating='post')
X_val_pad   = pad_sequences(val_sequences,   maxlen=max_len, padding='post', truncating='post')
X_test_pad  = pad_sequences(test_sequences,  maxlen=max_len, padding='post', truncating='post')

vocab_size = len(tokenizer.word_index) + 1

# 2. Tạo ma trận trọng số cho Embedding Layer từ Word2Vec

w2v_model = Word2Vec.load("lab5_2/models/word2vec.model")
embedding_dim = w2v_model.vector_size
embedding_matrix = np.zeros((vocab_size, embedding_dim))

found, total = 0, 0
for word, i in tokenizer.word_index.items():
    total += 1
    if word in w2v_model.wv:
        embedding_matrix[i] = w2v_model.wv[word]
        found += 1
oov_rate = 1 - found / total
print(f"Tỷ lệ từ không có trong Word2Vec: {oov_rate:.2%}")

# 3. Xây dựng mô hình Sequential với LSTM


lstm_model_pretrained = Sequential([
    Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        weights=[embedding_matrix],
        input_length=max_len,
        trainable=True  
    ),
    Bidirectional(LSTM(128, dropout=0.3, recurrent_dropout=0.2, return_sequences=False)),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(num_classes, activation='softmax')
])

lstm_model_pretrained.compile(
    optimizer=Adam(learning_rate=0.0008),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 4. Huấn luyện mô hình

es = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)

history = lstm_model_pretrained.fit(
    X_train_pad, y_train,
    validation_data=(X_val_pad, y_val),
    epochs=25,
    batch_size=16,
    callbacks=[es, rlr],
    verbose=1
)

# 5. Đánh giá mô hình

test_loss, test_acc = lstm_model_pretrained.evaluate(X_test_pad, y_test, verbose=0)
print(f"\nTest Accuracy: {test_acc:.4f} | Test Loss: {test_loss:.4f}")

y_pred = np.argmax(lstm_model_pretrained.predict(X_test_pad), axis=1)
report = classification_report(y_test, y_pred, zero_division=0)

print("\n=== Evaluation: Embedding Pre-trained + LSTM ===")
print(report)


# 6. Lưu mô hình, tokenizer, dữ liệu và kết quả

os.makedirs("lab5_2/models", exist_ok=True)
os.makedirs("lab5_2/results", exist_ok=True)

# Lưu mô hình và tokenizer
lstm_model_pretrained.save("lab5_2/models/lstm_pretrained.keras")
joblib.dump(tokenizer, "lab5_2/models/tokenizer.pkl")

# Lưu dữ liệu tiền xử lý cho NV4
np.save("lab5_2/models/X_train_pad.npy", X_train_pad)
np.save("lab5_2/models/X_val_pad.npy", X_val_pad)
np.save("lab5_2/models/X_test_pad.npy", X_test_pad)
np.save("lab5_2/models/y_train.npy", y_train)
np.save("lab5_2/models/y_val.npy", y_val)
np.save("lab5_2/models/y_test.npy", y_test)
np.savez("lab5_2/models/meta_info.npz",
         vocab_size=vocab_size,
         max_len=max_len,
         num_classes=num_classes)

# Lưu kết quả đánh giá
with open("lab5_2/results/nv3_res.txt", "w", encoding="utf-8") as f:
    f.write("=== Evaluation: Embedding Pre-trained + LSTM ===\n")
    f.write(f"Test Accuracy: {test_acc:.4f} | Test Loss: {test_loss:.4f}\n\n")
    f.write(report)

print("\n Kết quả và dữ liệu tiền xử lý đã được lưu vào: lab5_2/models/ và lab5_2/results/")

