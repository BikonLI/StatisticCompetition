import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import optuna
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from data import *

# 设置设备（使用 GPU 加速）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====================
# 1. 定义神经网络模型
# ====================
class RegressionNet(nn.Module):
    def __init__(self, input_size=385, hidden_size=128):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = 1
        self.file = "sld_nn/model.pth"

        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, self.output_size)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)  # 添加 Dropout 防止过拟合

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)  # 添加 Dropout
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def save(self):
        torch.save(self.state_dict(), self.file)
        
    def load(self):
        if os.path.exists(self.file):
            self.load_state_dict(torch.load(self.file))
            print("✅ 已加载模型权重")


# 归一化数据
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 转换为 Tensor
X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test, y_test = torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# 创建 DataLoader
batch_size = 64
train_dataset = Data.TensorDataset(X_train, y_train)
test_dataset = Data.TensorDataset(X_test, y_test)
train_loader = Data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = Data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ====================
# 3. 训练 & 评估函数
# ====================
def train_model(model, optimizer, criterion, train_loader, test_loader, epochs=50, patience=5):
    model.to(device)
    best_loss = float("inf")
    no_improve_epochs = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)  # 梯度裁剪
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # 验证集评估
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                predictions = model(batch_x)
                loss = criterion(predictions, batch_y)
                val_loss += loss.item()
        
        val_loss /= len(test_loader)

        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # 早停机制
        if val_loss < best_loss:
            best_loss = val_loss
            no_improve_epochs = 0
            model.save()
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print("⏹️  早停触发，训练停止！")
                break

    return best_loss

# ====================
# 4. 超参数优化（Optuna）
# ====================
def objective(trial):
    hidden_size = trial.suggest_int("hidden_size", 64, 512)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)

    # 创建模型
    model = RegressionNet(hidden_size=hidden_size).to(device)
    
    # 选择优化器
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    
    # 损失函数
    criterion = nn.MSELoss()
    
    # 训练模型
    val_loss = train_model(model, optimizer, criterion, train_loader, test_loader, epochs=50)
    
    return val_loss

# 运行 Optuna 进行超参数优化
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)

# 输出最佳参数
best_params = study.best_params
print(f"✅ 最佳超参数: {best_params}")

# ====================
# 5. 训练最终模型
# ====================
print("🚀 训练最终模型...")
best_model = RegressionNet(hidden_size=best_params["hidden_size"]).to(device)
best_optimizer = optim.AdamW(best_model.parameters(), lr=best_params["learning_rate"], weight_decay=1e-2)
best_criterion = nn.MSELoss()

train_model(best_model, best_optimizer, best_criterion, train_loader, test_loader, epochs=100)

print("🎉 训练完成！最佳模型已保存。")