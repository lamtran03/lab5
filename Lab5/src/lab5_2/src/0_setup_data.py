import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import tarfile

# 1. Giải nén 
data_path = "data/hwu.tar.gz"
extract_dir = "data/"

if os.path.exists(data_path):
    with tarfile.open(data_path, "r:gz") as tar:
        tar.extractall(path=extract_dir)
else:
    print("Bỏ qua bước giải nén (file đã có sẵn).")

# 2. Đọc dữ liệu 
train_path = "C:\\Users\\trand\\OneDrive\\Máy tính\\lab5-main\\lab5_2\\data\\hwu\\train.csv"
val_path   = "C:\\Users\\trand\\OneDrive\\Máy tính\\lab5-main\\lab5_2\\data\\hwu\\val.csv"
test_path  = "C:\\Users\\trand\\OneDrive\\Máy tính\\lab5-main\\lab5_2\\data\\hwu\\test.csv"

df_train = pd.read_csv("C:\\Users\\trand\\OneDrive\\Máy tính\\lab5-main\\lab5_2\\data\\hwu\\train.csv", sep=',', header=0, names=['text', 'intent'])
df_val   = pd.read_csv("C:\\Users\\trand\\OneDrive\\Máy tính\\lab5-main\\lab5_2\\data\\hwu\\val.csv", sep=',', header=0, names=['text', 'intent'])
df_test  = pd.read_csv("C:\\Users\\trand\\OneDrive\\Máy tính\\lab5-main\\lab5_2\\data\\hwu\\test.csv", sep=',', header=0, names=['text', 'intent'])


print("Đọc dữ liệu thành công:")
print("Train shape:", df_train.shape)
print("Validation shape:", df_val.shape)
print("Test shape:", df_test.shape)

print("\nMẫu dữ liệu huấn luyện:")
print(df_train.head())

# 3. Mã hóa nhãn intent thành số
encoder = LabelEncoder()
encoder.fit(pd.concat([df_train['intent'], df_val['intent'], df_test['intent']]))

df_train['label'] = encoder.transform(df_train['intent'])
df_val['label']   = encoder.transform(df_val['intent'])
df_test['label']  = encoder.transform(df_test['intent'])

print("\nĐã mã hóa nhãn (intent) thành số:")
print(df_train.head())

# 4. Lưu 
output_dir = "lab5_2\\data\\hwu"
df_train.to_csv(os.path.join(output_dir, "train_processed.csv"), index=False)
df_val.to_csv(os.path.join(output_dir, "val_processed.csv"), index=False)
df_test.to_csv(os.path.join(output_dir, "test_processed.csv"), index=False)

