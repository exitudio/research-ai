import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt


class Trainer():
    def __init__(self, env, num_episodes, num_mean_episode=10, save_name="checkpoint.pth"):
        self.env = env
        self.num_episodes = num_episodes
        self.num_mean_episode = num_mean_episode
        self.save_name = save_name

        # init variables
        self.mean_reward = deque(maxlen=self.num_mean_episode)
        self.best_reward = -1e8
        self.total_reward = []
        self.additional_log = {}

    def _save(self, reward): return 0
    def _load(self): return 0
    def _loop(self, episode) -> int: return 0
    def _policy(self, state): return 0

    def train(self, is_logged=True, early_stop=None):
        for episode in range(self.num_episodes):
            reward = self._loop(episode)
            self.total_reward.append(reward)
            if self.num_mean_episode:
                self.mean_reward.append(reward)
                current_mean_reward = np.mean(self.mean_reward)
                if is_logged:
                    print('\rEpisode {}\tAverage Score: {:.2f} \tother{}'.format(
                        episode+1, current_mean_reward, self.additional_log), end="")
                    if episode % self.num_mean_episode == self.num_mean_episode-1:
                        print('\rEpisode {}\tAverage Score: {:.2f} \tother{}'.format(
                            episode+1, current_mean_reward, self.additional_log))
                if current_mean_reward > self.best_reward:
                    self._save(reward)

                if early_stop and early_stop(current_mean_reward):
                    print('\r--- early stop ----')
                    print(' current_mean_reward:',
                          current_mean_reward, 'episode:', episode)
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
