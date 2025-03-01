import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from utils import RWN, get_rfweight

class CustomRWN(RWN):
    def __init__(self, hidden_size=256, device='cuda'):
        super().__init__(hidden_size, device)
        self.rf_model = None  # 存储随机森林模型
        self.file = "sld_rwn/model"

    def train(self, csv_path, tau=1e-3, batch_size=100, n_iter=2000, lr=1e-3, tol=1e-5, verbose=True):
        """
        训练 CustomRWN 模型
        """
        # 读取数据
        df = pd.read_csv(csv_path)
        x_train = df.iloc[:, 1:-1].values  # 取 value_i 作为输入
        y_train = df.iloc[:, -1].values  # 取 reference 作为目标
        
        # 训练随机森林
        params_rf = {'min_samples_split': [2, 3, 4, 5, 6, 7]}
        self.rf_model = RandomForestRegressor(n_estimators=100)
        reg_rf = GridSearchCV(self.rf_model, params_rf)
        reg_rf.fit(x_train, y_train)
        self.rf_model = reg_rf.best_estimator_
        self.rf_model.fit(x_train, y_train)
        
        # 计算随机森林权重
        rf_weights, _ = get_rfweight(self.rf_model, x_train)
        
        # 训练 RWN
        self.fit(x_train, y_train, rf_weights, tau, False, batch_size, n_iter, lr, tol, verbose)

    def save(self):
        """
        保存模型参数
        """
        torch.save(self.state_dict(), self.file + '_rwn.pth')
        joblib.dump(self.rf_model, self.file + '_rf.pkl')

    def load(self):
        """
        加载模型参数
        """
        self.load_state_dict(torch.load(self.file + '_rwn.pth'))
        self.rf_model = joblib.load(self.file + '_rf.pkl')

