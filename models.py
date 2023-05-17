import torch
import torch.nn as nn


# 定义单层感知机模型
class Perceptron(nn.Module):
    def __init__(self):
        super(Perceptron, self).__init__()
        self.fc = nn.Linear(2, 1)

    def forward(self, x):
        out = self.fc(x)
        return out


class PerceptronDouble(nn.Module):
    def __init__(self):
        super(PerceptronDouble, self).__init__()
        self.l1 = torch.nn.Linear(2, 2)
        self.l2 = torch.nn.Linear(2, 1)

    def forward(self, x):
        x = torch.sigmoid(self.l1(x))
        x = torch.sigmoid(self.l2(x))
        return x
