from .base import Base
import torch
import random
import numpy as np


class TableBase(Base):
    def __init__(self, env, num_episodes, epsilon, alpha, gamma, lambd=0):
        super().__init__(env, env.observation_space.n, env.action_space.n,
                         num_episodes, epsilon, alpha, gamma, lambd)
        self.Q = torch.zeros(self.num_state, self.num_action)

    def epsilon_greedy(self, state) -> int:
        if random.random() > self.epsilon:
            max_actions = self.Q[state] == self.Q[state].max()
            max_value = max_actions.nonzero().flatten()
            action = max_value[torch.randint(max_value.shape[0], (1,))].item()
        else:
            action = random.randrange(0, self.num_action)
        return action

    def convert_Q_to_V(self):
        V = np.array([0]*self.env.nS)
        for state, state_value in enumerate(self.Q.numpy()):
            V[state] = np.max(state_value)
        return V.reshape(self.env.shape)


class ApproximationBase(Base):
    def __init__(self, env, num_state, num_action, num_episodes, epsilon, alpha, gamma, lambd=0):
        super().__init__(
            env,
            num_state, 
            num_action,
            num_episodes,
            epsilon,
            alpha,
            gamma,
            lambd)

        hidden = 8
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.num_state * self.num_action, 1),
            # torch.nn.Linear(self.num_state * self.num_action, hidden),
            # torch.nn.ReLU(),
            # torch.nn.Linear(hidden, hidden),
            # torch.nn.ReLU(),
            # torch.nn.Linear(hidden, 1),
        )
        self.loss_fn = torch.nn.MSELoss(reduction='sum')
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=alpha)  # ???? learning rate???

    def approximate_q(self, state, action: int):
        return self.model(self.feature(state, action))

    def update_weight(self, td_target, approx_q):
        loss = self.loss_fn(td_target, approx_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def feature(self, state, action):
        self._feature = torch.zeros(self.num_action, self.num_state)
        self._feature[action] = torch.from_numpy(state).float()
        return self._feature.view(-1)

    def epsilon_greedy(self, state) -> int:
        if random.random() > self.epsilon:
            # Not support multiple max a
            action = self.argmax([self.approximate_q(state, a)
                                  for a in range(self.num_action)])
        else:
            action = random.randrange(0, self.num_action)
        return action

    @staticmethod
    def argmax(x):
        if isinstance(x, torch.Tensor):
            return x.max(dim=0)[1][0]
        elif isinstance(x, list):
            return x.index(max(x))
