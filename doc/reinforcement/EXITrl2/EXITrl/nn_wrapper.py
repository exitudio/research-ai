import torch
from .helpers import convert_to_tensor, device
import random
import numpy as np


class NNWrapper:
    def __init__(self, model, lr):
        self.model = model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def forward(self, *input):
        input = tuple(i.to(device) for i in input)
        output = self.model(*input)
        return output

    def backprop(self, loss):
        self.optimizer.zero_grad()
        loss.backward(retain_graph=False)
        self.optimizer.step()

    def epsilon_greedy(self, state, epsilon):
        self.model.eval() # notify model to use eval mode in dropout/batchnorm
        with torch.no_grad():
            action_values = self.forward(convert_to_tensor(state))
        self.model.train()
        if random.random() > epsilon:
            return np.argmax(action_values.cpu().numpy())
        else:
            return random.choice(np.arange(action_values.shape[0]))
