import torch
import torch.nn as nn

import models

# 定义训练数据, 这里做的是一个与运算
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[0], [0], [0], [1]], dtype=torch.float32)

train_set = torch.utils.data.TensorDataset(X, y)

# 定义批处理大小和是否打乱
batch_size = 64
shuffle = True

# 创建数据加载器
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)

# 定义单层感知机模型
model = models.Perceptron()

# 定义损失函数和优化器
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 训练模型
num_epochs = 10000
for epoch in range(num_epochs):
    for i, (input_data, output_data) in enumerate(train_loader):
        # 前向传播
        y_pred = model(input_data)

        # 计算损失
        loss = criterion(y_pred, output_data)

        # 反向传播并更新权重
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 打印损失
        if epoch % 100 == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}')

# 保存模型
torch.save(model.state_dict(), "and.pt")

# 测试模型
with torch.no_grad():
    y_pred = model(X)
    y_pred = torch.round(torch.sigmoid(y_pred))
    print(f'Predictions: {y_pred.flatten()}')
