import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import optuna
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import json
from data import *

# GPU 检测
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"ฅ^•ﻌ•^ฅ  喵喵~ 设备检测中... 使用的是 {'GPU✨' if torch.cuda.is_available() else 'CPU🔧'} 哦！")

# 定义神经网络
class RegressionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RegressionModel, self).__init__()
        print(f"✨ 乖乖~ 正在创建 {num_layers} 层的喵喵神经网络，隐藏层大小是 {hidden_size} 哦！")
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, 1))  # 输出层
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
    def save(self):
        torch.save(self.state_dict(), self.file)
        print("💾 喵呜！模型已经存好啦！")

    def load(self):
        if os.path.exists(self.file):
            self.load_state_dict(torch.load(self.file))
            print("✅ 已加载模型权重，喵~快来看看它变强了没！")

# 早停机制
class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.best_loss = float("inf")
        self.counter = 0
        print(f"🚀 早停机制启动！耐心值设为 {patience}，要是学习太久没有进步，喵就会喊停哦！")

    def step(self, loss):
        if loss < self.best_loss:
            self.best_loss = loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            print(f"🥺 喵呜…… 验证损失 {loss:.6f} 没有变好呢，已经 {self.counter}/{self.patience} 轮了喵！")
            return self.counter >= self.patience

# 超参数优化
def objective(trial):
    hidden_size = trial.suggest_int("hidden_size", 16, 128)
    num_layers = trial.suggest_int("num_layers", 1, 3)
    learning_rate = trial.suggest_loguniform("lr", 1e-4, 1e-2)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    print(f"✨ 开始喵喵试验 {trial.number}：hidden_size={hidden_size}, num_layers={num_layers}, lr={learning_rate:.6f}, batch_size={batch_size}")

    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = RegressionModel(input_size=X.shape[1], hidden_size=hidden_size, num_layers=num_layers).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    early_stopping = EarlyStopping(patience=10)

    # 训练模型
    for epoch in range(100):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                val_loss += criterion(outputs, batch_y).item()
        val_loss /= len(val_loader)

        print(f"📊 试验 {trial.number} - Epoch {epoch+1}: 验证损失 {val_loss:.6f}")

        if early_stopping.step(val_loss):
            print("💤 早停触发！喵不练了！")
            break

    return val_loss


BEST_PARAMS_FILE = "boston/best_hyperparams.json"
def save_best_hyperparams(params):
    """保存最优超参数到 JSON 文件"""
    with open(BEST_PARAMS_FILE, "w") as f:
        json.dump(params, f, indent=4)
    print("📁✨ 最优超参数已经存储到 'best_hyperparams.json' 啦！下次可以直接加载哦！(ฅ^ω^ฅ)")

def load_best_hyperparams():
    """从 JSON 文件加载最优超参数"""
    if os.path.exists(BEST_PARAMS_FILE):
        with open(BEST_PARAMS_FILE, "r") as f:
            params = json.load(f)
        print("🎉 已成功加载之前找到的最优超参数！不用再从零开始啦！(≧▽≦)ﾉ")
        return params
    else:
        print("⚠️ 没有找到存储的超参数哦，先来探索最优参数吧！(；´Д`)")
        return None


