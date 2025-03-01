import torch
import torch.nn as nn
import torch.optim as optim
import os

# 定义神经网络模型
class RegressionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.file = "sld_nn/model.pth"
        
        # 假设 parentId 和 value 拼接后总共是 385 个输入特征（parentId + 384 values）
        self.input_size = 385
        # 假设隐藏层的节点数
        self.hidden_size = 128
        # 输出是一个浮点数（回归任务）
        self.output_size = 1
        
        # 定义隐藏层和输出层
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)  # 输入到隐藏层
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)  # 隐藏层到隐藏层
        self.fc3 = nn.Linear(self.hidden_size, self.output_size)  # 隐藏层到输出层

        # 激活函数（使用ReLU）
        self.relu = nn.ReLU()

    def forward(self, x):
        # 前向传播
        x = self.relu(self.fc1(x))  # 通过第一个隐藏层
        x = self.relu(self.fc2(x))  # 通过第二个隐藏层
        x = self.fc3(x)  # 输出层
        return x
    # 假设模型是 model
    
    def save(self):
        torch.save(self.state_dict(), self.file)  # 保存模型的状态字典
        
        
    def load(self):
        self.load_state_dict(torch.load(self.file))


# 创建网络实例
model = RegressionNet()
if os.path.exists(model.file):
    model.load()                    # 如果存在模型文件，那么直接加载。

# 打印模型结构
print(model)
