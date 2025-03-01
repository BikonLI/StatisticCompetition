import torch
from sklearn.metrics import mean_squared_error, r2_score
from model import *
from data import *
import numpy as np

# 假设 X_test 和 y_test 是你测试集的数据和标签
# 将数据转换为张量，并使用 CPU

# 使模型处于评估模式
model.eval()

# 使用模型进行预测
with torch.no_grad():  # 在推理时不需要计算梯度
    y_pred = model(X_test)  # 获取模型预测结果

# 计算 R² 分数
y_pred = y_pred.numpy()  # 将预测值转换为 NumPy 数组
y_test_tensor = y_test.numpy()  # 将真实值转换为 NumPy 数组

r2 = r2_score(y_test_tensor, y_pred)  # 计算 R² 分数
print(f'R² Score: {r2:.4f}')

# 计算 MSE（均方误差）
mse = mean_squared_error(y_test_tensor, y_pred)  # 计算均方误差
print(f'Mean Squared Error (MSE): {mse:.4f}')

# 可选：计算 RMSE（均方根误差）
rmse = np.sqrt(mse)  # 计算均方根误差
print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')
