import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 读取本地 CSV 文件（假设文件名为 data.csv，最后一列是目标值）
df = pd.read_csv("boston.csv")

# 假设最后一列是目标变量，前面的列是特征
X = df.iloc[:, :-1].values  # 特征
y = df.iloc[:, -1].values.reshape(-1, 1)  # 目标变量（回归任务）

# 归一化特征和目标值
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y)

# 训练集和测试集划分
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 转换为 PyTorch Tensor
X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)
X_val, y_val = torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)

train_dataset = data.TensorDataset(X_train, y_train)
val_dataset = data.TensorDataset(X_val, y_val)
