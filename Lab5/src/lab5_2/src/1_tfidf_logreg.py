import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
import pandas as pd



# 1. Đọc dữ liệu
train_path = "lab5_2\\data\\hwu\\train_processed.csv"
val_path   = "lab5_2\\data\\hwu\\val_processed.csv"
test_path  = "lab5_2\\data\\hwu\\test_processed.csv"



df_train = pd.read_csv(train_path)
df_val   = pd.read_csv(val_path)
df_test  = pd.read_csv(test_path)

y_train = df_train["intent"]
y_test = df_test["intent"]


# 2. Tạo pipeline TF-IDF + Logistic Regression
tfidf_lr_pipeline = make_pipeline(
    TfidfVectorizer(max_features=5000),
    LogisticRegression(max_iter=1000)
)

# 3. Train mô hình trên tập train
tfidf_lr_pipeline.fit(df_train["text"], df_train["label"])

# 4. Đánh giá trên tập test
y_pred = tfidf_lr_pipeline.predict(df_test["text"])
print("\nEvaluation on test set:")
print(classification_report(df_test["label"], y_pred))

# 5. Lưu mô hình và kết quả
import joblib
joblib.dump(tfidf_lr_pipeline, "lab5_2/models/tfidf_logreg.pkl")

results_dir = "lab5_2/results"
os.makedirs(results_dir, exist_ok=True)

report_text = classification_report(df_test["label"], y_pred)
with open(os.path.join(results_dir, "nv1_res.txt"), "w", encoding="utf-8") as f:
    f.write("=== Evaluation: TF-IDF + Logistic Regression ===\n")
    f.write(report_text)

print(f"\nKết quả đã được lưu vào: {os.path.join(results_dir, 'nv1_res.txt')}")



