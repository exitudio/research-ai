import torch
from .helpers import convert_to_tensor


class NNWrapper:
    def __init__(self, model, lr):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def forward(self, *input):
        output = self.model(*input)
        return output

    def backprop(self, loss):
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()
