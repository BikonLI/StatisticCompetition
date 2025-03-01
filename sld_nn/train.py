# 训练模型
from model import * # 加载模型
from data import *  # 加载数据

# 假设我们有训练数据和标签
# 假设 `X_train` 是一个形状为 (N, 385) 的张量，表示 N 个样本，每个样本有 385 个特征
# `y_train` 是一个形状为 (N, 1) 的张量，表示每个样本的参考值

# 选择优化器和损失函数
criterion = nn.MSELoss()  # 回归任务使用均方误差损失
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam优化器

# 假设我们有训练数据
# X_train, y_train 是你的训练数据
# 例如：X_train = torch.randn(100, 385), y_train = torch.randn(100, 1)

num_epochs = 500  # 训练轮数
for epoch in range(num_epochs):
    model.train()  # 设置模型为训练模式

    # 前向传播
    outputs = model(X_train)  # 获取预测值
    loss = criterion(outputs, y_train)  # 计算损失

    # 反向传播和优化
    optimizer.zero_grad()  # 清空梯度
    loss.backward()  # 计算梯度
    optimizer.step()  # 更新模型参数

    if (epoch + 1) % 10 == 0:  # 每10轮打印一次损失
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        model.save()
