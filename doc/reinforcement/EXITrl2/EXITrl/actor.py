import torch
from .helpers import device, convert_to_tensor
import torch.nn.functional as F


class Actor():
    def __init__(self, model=None, lr=0.01):
        self.model = model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

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