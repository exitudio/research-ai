class Base():
    def __init__(self, env, num_state, num_action, num_episodes, epsilon, alpha, gamma, lambd):
        self.env = env
        self.num_episodes = num_episodes
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.lambd = lambd

        self.num_state = num_state
        self.num_action = num_action

    def train(self, is_logged=False):
        total_reward = []
        for episode in range(self.num_episodes):
            reward = self._loop(episode)
            total_reward.append(reward)
            if is_logged:
                print('episode:', episode, 'reward:', reward)

    def _loop(self, episode) -> int: return 0
