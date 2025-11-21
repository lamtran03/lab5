import os
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report

# === 1. Load dữ liệu và tokenizer từ NV3 ===
print("Đang tải dữ liệu tiền xử lý từ NV3...")

tokenizer = joblib.load("lab5_2/models/tokenizer.pkl")

X_train_pad = np.load("lab5_2/models/X_train_pad.npy")
X_val_pad   = np.load("lab5_2/models/X_val_pad.npy")
X_test_pad  = np.load("lab5_2/models/X_test_pad.npy")
y_train     = np.load("lab5_2/models/y_train.npy")
y_val       = np.load("lab5_2/models/y_val.npy")
y_test      = np.load("lab5_2/models/y_test.npy")

meta = np.load("lab5_2/models/meta_info.npz")
vocab_size = int(meta["vocab_size"])
max_len = int(meta["max_len"])
num_classes = int(meta["num_classes"])

print(f" Loaded preprocessed data | vocab_size={vocab_size}, max_len={max_len}, classes={num_classes}")

# === 2. Xây dựng mô hình (Embedding học từ đầu) ===
embedding_dim = 100

lstm_model_scratch = Sequential([
    Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        input_length=max_len
    ),
    Bidirectional(LSTM(128, dropout=0.3, recurrent_dropout=0.2)),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(num_classes, activation='softmax')
])

lstm_model_scratch.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# === 3. Huấn luyện ===
es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)

history = lstm_model_scratch.fit(
    X_train_pad, y_train,
    validation_data=(X_val_pad, y_val),
    epochs=30,
    batch_size=32,
    callbacks=[es, rlr],
    verbose=1
)

# === 4. Đánh giá ===
test_loss, test_acc = lstm_model_scratch.evaluate(X_test_pad, y_test, verbose=0)
print(f"\nTest Accuracy: {test_acc:.4f} | Test Loss: {test_loss:.4f}")

y_pred = np.argmax(lstm_model_scratch.predict(X_test_pad), axis=1)
report = classification_report(y_test, y_pred, zero_division=0)
print("\n=== Evaluation: Embedding học từ đầu + LSTM ===")
print(report)

# === 5. Lưu kết quả ===
os.makedirs("lab5_2/results", exist_ok=True)
with open("lab5_2/results/nv4_res.txt", "w", encoding="utf-8") as f:
    f.write("=== Evaluation: Embedding học từ đầu + LSTM ===\n")
    f.write(f"Test Accuracy: {test_acc:.4f} | Test Loss: {test_loss:.4f}\n\n")
    f.write(report)

print("\n Kết quả NV4 đã được lưu vào: lab5_2/results/nv4_res.txt")
