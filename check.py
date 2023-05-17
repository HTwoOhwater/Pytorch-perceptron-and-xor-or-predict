import torch

import models

Perceptron = models.PerceptronDouble()
Perceptron.load_state_dict(torch.load("xor.pt"))

params = Perceptron.state_dict()
print(params)
