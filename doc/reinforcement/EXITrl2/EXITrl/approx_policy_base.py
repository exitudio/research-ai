from .base import Base
from .helpers import convert_to_tensor
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
        self.approx_policy['optimizer'] = torch.optim.Adam(
            self.approx_policy['model'].parameters(), lr=self.__dict__[learning_rate_name])

    def update_policy(self, loss):
        self.approx_policy['optimizer'].zero_grad()
        loss.backward(retain_graph=True)
        self.approx_policy['optimizer'].step()

    def softmax_policy(self, state):
        state = convert_to_tensor(state)
        output = self.approx_policy['model'](state)
        probs = F.softmax(output, dim=0)

        # 1. Categorical
        # m = Categorical(probs)
        # action = m.sample()
        # return action.item(), m.log_prob(action)

        # 2. manual
        action = torch.multinomial(probs, num_samples=1)[0]  # np.choice
        log_prob = torch.log(probs[action])
        return action.item(), log_prob

    def gaussian_policy(self, state):
        state = convert_to_tensor(state)
        mu, sigma = self.approx_policy['model'](state)
        sigma = sigma+1e-8  # add small number to avoid 0
        m = torch.distributions.MultivariateNormal(
            mu, torch.eye(self.num_action)*sigma)
        action = m.sample()
        action = torch.clamp(
            action,
            # TODO clip each element separately, not the whole vector with same number
            self.env.action_space.low[0],
            self.env.action_space.high[0])
        return action.numpy(), m.log_prob(action), m.entropy()

    def print_params(self):
        for name, param in self.approx_policy['model'].named_parameters():
            if param.requires_grad:
                print(name, param.data)
