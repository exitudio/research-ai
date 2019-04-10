from .base import Base
import torch
import random
import numpy as np
import numbers


class TableBase(Base):
    def initialize(self, learning_rate_name="alpha"):
        self.Q = torch.zeros(*self.state_shape, self.num_action)
        self.learning_rate_name = learning_rate_name

    def get_q(self, state):
        if isinstance(state, numbers.Number):
            return self.Q[state]
        else:
            return self.Q[tuple(state)]
        

    def update_q(self, td_target, state, action):
        if isinstance(state, numbers.Number):
            query = (state, action)
        else:
            query = (*state, action)
        current_q = self.get_q(state)[action]
        self.Q[query] += self.__dict__[self.learning_rate_name] * \
            (td_target - current_q)

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
