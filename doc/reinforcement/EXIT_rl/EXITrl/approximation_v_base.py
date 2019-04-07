from .base import Base
import torch
import random
import numpy as np


class ApproximationVBase(Base):
    def __init__(self, env, num_state, num_action, num_episodes, policy, epsilon, alpha, gamma, lambd=0):
        super().__init__(
            env,
            num_state,
            num_action,
            num_episodes,
            policy,
            epsilon,
            alpha,
            gamma,
            lambd)

    def initialize(self, model=None):
        self.approx_v = {}
        if model == None:
            hidden = 64
            self.approx_v['model'] = torch.nn.Sequential(
                torch.nn.Linear(self.num_state, hidden),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden, self.num_action),
            ).to(self.device)
        else:
            self.approx_v['model'] = model
        self.approx_v['loss_fn'] = torch.nn.MSELoss(reduction='sum')
        self.approx_v['optimizer'] = torch.optim.Adam(
            self.approx_v['model'].parameters(), lr=self.alpha)

    def approximate_q(self, state):
        state = torch.from_numpy(state).float()
        return self.approx_v['model'](state)

    def update_weight(self, td_target, approx_q):
        loss = self.approx_v['loss_fn'](td_target, approx_q)
        self.approx_v['optimizer'].zero_grad()
        loss.backward()
        self.approx_v['optimizer'].step()

    def epsilon_greedy(self, state) -> int:
        if random.random() > self.epsilon:
            # Not support multiple max a
            qs = self.approximate_q(state)
            action = qs.argmax().item()
        else:
            action = random.randrange(0, self.num_action)
        return action

    def print_params(self):
        for name, param in self.approx_v['model'].named_parameters():
            if param.requires_grad:
                print(name, param.data)


class ExperienceReplay:
    '''
        TODO random getting batch experience rather than use the whole
    '''

    def __init__(self, num_experience=128, batch_size=32):
        self.num_experience = num_experience
        self.memory = []
        self.batch_size = batch_size

    def remember(self, state, action, reward, state_, action_, done):
        i = 0
        for i in range(len(self.memory)):
            if self.memory[i][2] < reward:
                break
        self.memory.insert(i, (state, action, reward, state_,
                               action_, done))  # sort reward
        if len(self.memory) > self.num_experience:
            del self.memory[0]

    def get_batch(self, get_target):
        targets, predict_qs = None, None
        for _, data in enumerate(self.memory):
            target, predict_q = get_target(*data)
            if targets is None:
                targets = torch.empty([0, *target.shape], requires_grad=True)
                predict_qs = torch.empty(
                    [0, *predict_q.shape], requires_grad=True)
            targets = torch.cat((targets, target.view([1, *target.shape])))
            predict_qs = torch.cat(
                (predict_qs, predict_q.view([1, *predict_q.shape])))
        return torch.Tensor(targets), torch.Tensor(predict_qs)
