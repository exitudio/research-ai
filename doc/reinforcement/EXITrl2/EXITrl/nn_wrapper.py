import torch
from .helpers import convert_to_tensor, device
import random
import numpy as np
import torch.nn.functional as F


class NNWrapper:
    def __init__(self, model, lr):
        self.model = model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def forward(self, *input):
        input = tuple( convert_to_tensor(i) for i in input)
        return self.model(*input)

    def softmax_policy(self, state):
        state = convert_to_tensor(state)
        output = self.model(state)
        probs = F.softmax(output, dim=0)

        # 1. Categorical
        # m = Categorical(probs)
        # action = m.sample()
        # return action.item(), m.log_prob(action)

        # 2. manual
        action = torch.multinomial(probs, num_samples=1)[0]  # np.choice
        log_prob = torch.log(probs[action])
        return action.item(), log_prob

    def update(self, loss):
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
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
