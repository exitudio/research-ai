from .base import Base
import torch
import random
import numpy as np


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

        hidden = 64
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.num_state, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, self.num_action),
        )
        self.loss_fn = torch.nn.MSELoss(reduction='sum')
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=alpha)  # ???? learning rate???

    def approximate_q(self, state):
        state = torch.from_numpy(state).float()
        return self.model(state)

    def update_weight(self, td_target, approx_q):
        loss = self.loss_fn(td_target, approx_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def epsilon_greedy(self, state) -> int:
        if random.random() > self.epsilon:
            # Not support multiple max a
            qs = self.approximate_q(state)
            action = qs.argmax().item()
        else:
            action = random.randrange(0, self.num_action)

        # if (np.array_equal(state, [0, 0]) or np.array_equal(state, [0, 1]) or np.array_equal(state, [0, 2])):
        #     action = 1
        # elif (np.array_equal(state, [0, 3]) or np.array_equal(state, [1, 0]) or np.array_equal(state, [1, 1]) or np.array_equal(state, [1, 2]) or np.array_equal(state, [1, 3])):
        #     action = 2
        # elif (np.array_equal(state, [2, 0]) or np.array_equal(state, [2, 1]) or np.array_equal(state, [2, 2]) or np.array_equal(state, [2, 3])):
        #     action = 2
        # elif (np.array_equal(state, [3, 0]) or np.array_equal(state, [3, 1]) or np.array_equal(state, [3, 2]) or np.array_equal(state, [3, 3])):
        #     action = 1
        return action

    def print_params(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name, param.data)


