import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from data import *
import joblib
import os

# 假设 data 的列是 ['parentId', 'value1', 'value2', ..., 'value384', 'reference']

# 若模型存在，则加载，否则创建。
existance = os.path.exists("sld_rf/model.pkl")
rf_model = joblib.load('random_forest_model.pkl') if existance else RandomForestRegressor(n_estimators=10, random_state=42)

# 训练模型
rf_model.fit(X_train, y_train)
joblib.dump(rf_model, 'sld_rf/model.pkl')

# 使用测试集进行预测
y_pred = rf_model.predict(X_test)

# 评估模型
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)  # 均方误差
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f'R²: {r2:.4f}')
print(f'Mean Squared Error: {mse}')
print(f'RMSE: {rmse:.4f}')
