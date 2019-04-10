from .base import Base
import torch
import random
import numpy as np
import torch.nn.functional as F
from torch.distributions import Categorical


class ApproxPolicyBase(Base):
    def initialize(self, model=None, learning_rate_name="alpha"):
        self.approx_policy = {}
        if model == None:
            hidden = 8
            self.approx_policy['model'] = torch.nn.Sequential(
                torch.nn.Linear(self.num_state, hidden),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden, self.num_action),
            ).to(self.device)
        else:
            self.approx_policy['model'] = model
        self.approx_policy['loss_fn'] = torch.nn.MSELoss(reduction='sum')
        self.approx_policy['optimizer'] = torch.optim.Adam(
            self.approx_policy['model'].parameters(), lr=self.__dict__[learning_rate_name])

    def update_policy(self, loss):
        self.approx_policy['optimizer'].zero_grad()
        loss.backward(retain_graph=True)
        self.approx_policy['optimizer'].step()

    def softmax_policy(self, state):
        if type(state) is np.ndarray:
            state = torch.from_numpy(state).float()
        else:
            state = torch.FloatTensor([state])

        features = self.approx_policy['model'](state)
        probs = F.softmax(features, dim=0)

        # 1. Categorical
        # m = Categorical(probs)
        # action = m.sample()
        # return action.item(), m.log_prob(action)

        # 2. manual
        action = torch.multinomial(probs, 1)[0] # np.choice
        log_prob = torch.log(probs[action])
        return action.item(), log_prob

        

    def print_params(self):
        for name, param in self.approx_policy['model'].named_parameters():
            if param.requires_grad:
                print(name, param.data)
