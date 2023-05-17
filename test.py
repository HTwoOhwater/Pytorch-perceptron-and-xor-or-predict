import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score

torch.manual_seed(1234)

n = 500
x = torch.rand(n, 2) * 2 - 1
y = torch.Tensor([[1, 0] if k[0] * k[1] > 0 else [0, 1] for k in x])


class mm_cls(torch.nn.Module):
    def __init__(self):
        super(mm_cls, self).__init__()
        self.l1 = torch.nn.Linear(2, 5)
        self.l2 = lambda x: x ** 2
        self.l3 = torch.nn.Linear(5, 2)
        self.l4 = nn.Softmax(dim=1)

    def forward(self, x):
        x1 = self.l1(x)
        x2 = self.l2(x1)
        x3 = self.l3(x2)
        x4 = self.l4(x3)
        return x4


mm = mm_cls()

loss_fun = torch.nn.MSELoss()
optimizer = torch.optim.Adam(mm.parameters(), lr=0.001)

epochs = 2000
for i in range(epochs):
    y_hat = mm.forward(x)

    loss = loss_fun(y_hat, y)
    loss.backward()
    optimizer.step()
    if i % 100 == 0:
        print('step = %4d' % i, 'loss = %.4e' % loss.item())

    optimizer.zero_grad

y_hat = y_hat[:, 0] < y_hat[:, 1]
y = y[:, 0] < y[:, 1]

acc = accuracy_score(y_hat, y)
print('acc = %f' % acc)

import matplotlib.pyplot as plt

plt.subplot(121)
plt.plot(x[y == 1, 0], x[y == 1, 1], 'om')
plt.plot(x[y == 0, 0], x[y == 0, 1], 'ob')
plt.title('original')

plt.subplot(122)
plt.lot(x[y_hat == 1, 0], x[y_hat == 1, 1], 'om')
plt.plot(x[y_hat == 0, 0], x[y_hat == 0, 1], 'ob')
plt.title('predicted')
