import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("slice_localization_data.csv")  # 替换成你数据的路
X = data.iloc[:, 1:-1].values  # 特征部分，假设第1到倒数第二列是特征
y = data.iloc[:, -1].values  # 目标部分，最后一列是 reference

# 拆分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
