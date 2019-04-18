import torch
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from gridworld_env_2d_state import GridworldEnv2DState
import numpy as np


class Base():
    def __init__(self, env, num_episodes, policy, gamma=.9, alpha=.5, beta=.5, lambd=0, epsilon=.1):
        self.env = env
        self.num_episodes = num_episodes
        self._policy = policy
        self.epsilon = epsilon
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.lambd = lambd

        # state
        if isinstance(env.observation_space, Discrete):
            self.state_shape = [env.observation_space.n]
        elif isinstance(env.observation_space, Box):
            self.num_state = self.env.observation_space.shape[0]

        # action
        if isinstance(env.action_space, Discrete):
            self.num_action = self.env.action_space.n
        elif isinstance(env.action_space, Box):
            self.num_action = self.env.action_space.shape[0]

        # Duct tape
        if isinstance(env, GridworldEnv2DState):
            """This is Ducttape"""
            # stat_shape is using only in Discrete env
            self.state_shape = [4, 4]
            self.num_state = 2
            self.num_action = self.env.action_space.n

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def train(self, is_logged=False):
        total_reward = []
        for episode in range(self.num_episodes):
            reward = self._loop(episode)
            total_reward.append(reward)
            if is_logged:
                print('episode:', episode, 'reward:', reward)

    def _loop(self, episode) -> int: return 0

    def policy(self, state) -> int:
        """
        epsilon greedy method
        :return: action (int)
        """
        return getattr(self, self._policy)(state)

