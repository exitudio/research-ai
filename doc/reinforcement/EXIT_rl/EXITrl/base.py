import torch


class Base():
    def __init__(self, env, num_state, num_action, num_episodes, policy, epsilon, alpha, gamma, lambd):
        self.env = env
        self.num_episodes = num_episodes
        self._policy = policy
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.lambd = lambd

        self.num_state = num_state
        self.num_action = num_action

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
