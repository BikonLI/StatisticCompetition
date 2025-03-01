from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from data import *
from utils import *
import os

rwn_model = RWN(hs=64, device='cpu')  # 若你有 GPU，可以将 device 改为 'cuda'
if os.path.exists(rwn_model.file):
    rwn_model.load()
# 预测
y_pred = rwn_model.predict(X_test)

# 计算 R2 和 RMSE
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# 输出评估指标
print(f"R2 Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")