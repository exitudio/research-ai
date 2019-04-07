from .base import Base
import torch
import random
import numpy as np


class TableBase(Base):
    def __init__(self, env, num_episodes, policy, epsilon, alpha, gamma, lambd=0):
        super().__init__(env, env.observation_space.n, env.action_space.n,
                         num_episodes, policy, epsilon, alpha, gamma, lambd)
        self.Q = torch.zeros(self.num_state, self.num_action)

    def epsilon_greedy(self, state) -> int:
        if random.random() > self.epsilon:
            max_actions = self.Q[state] == self.Q[state].max()
            max_value = max_actions.nonzero().flatten()
            action = max_value[torch.randint(max_value.shape[0], (1,))].item()
        else:
            action = random.randrange(0, self.num_action)

        # if state in (0,1,2):
        #     action = 1
        # elif state in (3, 4,5,6,7):
        #     action = 2
        # elif state in (8,9,10,11):
        #     action = 2
        # elif state in (12,13,14,15):
        #     action = 1

        return action

    def convert_Q_to_V(self):
        V = np.array([0]*self.env.nS)
        for state, state_value in enumerate(self.Q.numpy()):
            V[state] = np.max(state_value)
        return V.reshape(self.env.shape)
