import torch
from .helpers import device


class Critic():
    def __init__(self, model=None, lr=0.01):
        self.model = model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def get_value(self, state):
        state = torch.from_numpy(state).float()
        return self.model(state)

    def update(self, loss):
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()
