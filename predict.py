import torch

import models

# 定义输入和输出维度
input_dim = 2
output_dim = 1
"""
# 定义单层感知机模型
model = models.Perceptron(input_dim, output_dim)
model.load_state_dict(torch.load("or.pt"))
"""
# 定义双层感知机模型
model = models.PerceptronDouble()
model.load_state_dict(torch.load("xor.pt"))

while True:
    a, b = input("输入一个预测数对\n").split()
    a = float(a)
    b = float(b)
    if a == 114514 and b == 1919810:
        break

    # 对输入数据进行预处理，需要将维度改为 (1, 2)
    x = torch.tensor([[a, b]], dtype=torch.float32)

    # 模型预测
    y_pred = torch.relu(torch.round(model(x)))
    y_pred[y_pred >= 0.5] = 1  # 设置接近1的值为1
    print(f"预测结果为：{int(y_pred.item())}")
