import torch
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from gridworld_env_2d_state import GridworldEnv2DState
import numpy as np
from collections import deque


class Base():
    def __init__(self, env, num_episodes, num_mean_episode=10, policy="", gamma=.9, alpha=.5, beta=.5, lambd=0, epsilon=.1, tau=1e-3, save_name="checkpoint.pth"):
        self.env = env
        self.num_episodes = num_episodes
        self.num_mean_episode = num_mean_episode
        self._policy = policy
        self.epsilon = epsilon
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.lambd = lambd
        self.tau = tau
        self.save_name = save_name

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

        # reward
        self.mean_reward = deque(maxlen=self.num_mean_episode)
        self.best_reward = -1e8
            
        self.total_reward = []
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def train(self, is_logged=True, early_stop=None):
        for episode in range(self.num_episodes):
            reward = self._loop(episode)
            self.total_reward.append(reward)
            if self.num_mean_episode:
                self.mean_reward.append(reward)
                current_mean_reward = np.mean(self.mean_reward)
                if is_logged:
                    print('\rEpisode {}\tAverage Score: {:.2f}'.format(
                        episode+1, current_mean_reward), end="")
                    if episode % self.num_mean_episode == self.num_mean_episode-1:
                        print('\rEpisode {}\tAverage Score: {:.2f}'.format(
                            episode+1, current_mean_reward))
                if current_mean_reward > self.best_reward:
                    self._save(reward)
                    
                if early_stop and early_stop(current_mean_reward):
                    print('\r--- early stop ----')
                    print(' current_mean_reward:', current_mean_reward, 'episode:', episode)
                    return
                    
                
    def _save(self, reward): return 0
    def _load(self): return 0                
    def _loop(self, episode) -> int: return 0
    
    def play(self, num_episode=3):
        self._load()
        for i in range(num_episode):
            state = self.env.reset()
            for j in range(1000):
                action = self.policy(state)
                self.env.render()
                state, reward, done, _ = self.env.step(action)
                if done: break 

    def policy(self, state) -> int:
        """
        epsilon greedy method
        :return: action (int)
        """
        return getattr(self, self._policy)(state)
