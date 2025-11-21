import sys
sys.stdout.reconfigure(encoding='utf-8')


import numpy as np
import pandas as pd
from gensim.models import Word2Vec
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
import joblib
import os

# 1. Đọc dữ liệu
train_path = "lab5_2/data/hwu/train_processed.csv"
val_path   = "lab5_2/data/hwu/val_processed.csv"
test_path  = "lab5_2/data/hwu/test_processed.csv"

df_train = pd.read_csv(train_path)
df_val   = pd.read_csv(val_path)
df_test  = pd.read_csv(test_path)

# Tách text thành danh sách từ
sentences = [text.split() for text in df_train["text"]]

# 2. Huấn luyện mô hình Word2Vec
w2v_model = Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=1,
    workers=4
)

# 3. Hàm chuyển mỗi câu thành vector trung bình
def sentence_to_avg_vector(text, model, vector_size=100):
    words = text.split()
    word_vecs = [model.wv[w] for w in words if w in model.wv]
    if len(word_vecs) == 0:
        return np.zeros(vector_size)
    return np.mean(word_vecs, axis=0)

# 4. Tạo dữ liệu vector trung bình 
X_train_avg = np.array([sentence_to_avg_vector(t, w2v_model) for t in df_train["text"]])
X_val_avg   = np.array([sentence_to_avg_vector(t, w2v_model) for t in df_val["text"]])
X_test_avg  = np.array([sentence_to_avg_vector(t, w2v_model) for t in df_test["text"]])

y_train = df_train["label"].values
y_val   = df_val["label"].values
y_test  = df_test["label"].values

num_classes = len(np.unique(y_train))

# 5. Xây dựng mô hình Dense
model = Sequential([
    Dense(128, activation='relu', input_shape=(w2v_model.vector_size,)),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 6. Huấn luyện mô hình
es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = model.fit(
    X_train_avg, y_train,
    validation_data=(X_val_avg, y_val),
    epochs=20,
    batch_size=32,
    callbacks=[es],
    verbose=1
)

# 7. Đánh giá trên tập test
test_loss, test_acc = model.evaluate(X_test_avg, y_test, verbose=0)
print(f"\nTest Accuracy: {test_acc:.4f} | Test Loss: {test_loss:.4f}")

# 8. In classification report
y_pred = np.argmax(model.predict(X_test_avg), axis=1)
print("\nEvaluation on test set:")
print(classification_report(y_test, y_pred, zero_division=0))

# 9. Lưu mô hình và kết quả
os.makedirs("models", exist_ok=True)
model.save("lab5_2/models/word2vec_dense.keras")
w2v_model.save("lab5_2/models/word2vec.model")
joblib.dump(w2v_model.wv.index_to_key, "lab5_2/models/vocab.pkl")

results_dir = "lab5_2/results"
os.makedirs(results_dir, exist_ok=True)

report_text = classification_report(y_test, y_pred)
with open(os.path.join(results_dir, "nv2_res.txt"), "w", encoding="utf-8") as f:
    f.write("=== Evaluation: Word2Vec (Average) + Dense Layer ===\n")
    f.write(f"Test Accuracy: {test_acc:.4f} | Test Loss: {test_loss:.4f}\n\n")
    f.write(report_text)

print(f"\nKết quả đã được lưu vào: {os.path.join(results_dir, 'nv2_res.txt')}")




