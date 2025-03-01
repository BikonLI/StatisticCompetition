# 读取csv
import torch
import pandas as pd
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 假设数据是 CSV 格式，读取数据
data = pd.read_csv("slice_localization_data.csv")

# 假设列名分别是 parentId, value1, value2, ..., value384, reference
X = data.iloc[:, :-1].values  # 所有列除了最后一列作为输入特征
y = data.iloc[:, -1].values   # 最后一列作为目标值

# 将数据转换为 PyTorch 张量
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)  # 需要转为列向量

X_np = X_tensor.numpy()  # 如果是在 GPU 上，需要先调用 .cpu()
y_np = y_tensor.numpy()  # 如果是在 GPU 上，需要先调用 .cpu()

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_np, y_np, test_size=0.2, random_state=42)
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

X_train = X_train.to(device)
X_test = X_test.to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)