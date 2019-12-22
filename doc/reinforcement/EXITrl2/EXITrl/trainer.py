import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from .helpers import MeanBuffer


class Trainer():
    def __init__(self, env, num_episodes, save_name="checkpoint.pth"):
        self.env = env
        self.num_episodes = num_episodes
        self.save_name = save_name

        # init variables
        self.best_reward = -1e8
        self.total_reward = []
        self.additional_log = {}

    def _save(self, reward): return 0
    def _load(self): return 0
    def _loop(self, episode) -> int: return 0
    def _policy(self, state): return 0

    def train(self, is_logged=True, early_stop=None, num_mean=10):
        mean_rewards = MeanBuffer(num_mean)
        for episode in range(self.num_episodes):
            reward = self._loop(episode)
            self.total_reward.append(reward)
            mean_rewards.add(reward)
            if is_logged:
                print('\rEpisode {}\tLast reward: {:.2f}\tAverage reward: {:.2f} \tother{}'.format(
                    episode+1, reward, mean_rewards.mean(), self.additional_log), end="")
                if episode % num_mean == num_mean-1:
                    print('\rEpisode {}\tLast reward: {:.2f}\tAverage reward: {:.2f} \tother{}                    '.format(
                        episode+1, reward, mean_rewards.mean(), self.additional_log))
            if mean_rewards.mean() > self.best_reward:
                self._save(reward)

            if early_stop and early_stop(mean_rewards.mean()):
                print('\r--- early stop ----                                                                            ')
                print('\rEpisode {}\tLast reward: {:.2f}\tAverage reward: {:.2f} \tother{}                    '.format(
                        episode+1, reward, mean_rewards.mean(), self.additional_log))
                return

    def play(self, num_episode=3):
        self._load()
        for episode in range(num_episode):
            state = self.env.reset()
            total_reward = 0
            for _ in range(1000):
                action = self._policy(state)
                self.env.render()
                state, reward, done, _ = self.env.step(action)
                total_reward += reward
                if done:
                    break
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(
                episode+1, total_reward))

    def plot_rewards(self):
        fig = plt.figure()
        fig.add_subplot(111)
        plt.plot(np.arange(1, len(self.total_reward)+1), self.total_reward)
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.show()
