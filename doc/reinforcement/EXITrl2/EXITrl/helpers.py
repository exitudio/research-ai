from collections import deque
import random
import torch
import numpy as np
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_simple_model(num_input, num_hidden, num_output):
    return torch.nn.Sequential(
        torch.nn.Linear(num_input, num_hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(num_hidden, num_output)
    )


def print_weight_size(model):
    for name, param in model.named_parameters():
        if param.requires_grad and "weight" in name:
            print(name, param.data.shape)


def copy_params(from_model, to_model):
    for from_param, to_param in zip(from_model.parameters(), to_model.parameters()):
        to_param.data.copy_(from_param.data)


def update_params(from_model, to_model, tau):
    for from_param, to_param in zip(from_model.parameters(), to_model.parameters()):
        update_data = tau*from_param.data + (1-tau) * to_param.data
        to_param.data.copy_(update_data)


def convert_to_tensor(input):
    if type(input) is np.ndarray:
        input = torch.from_numpy(input).float().to(device)
    else:
        input = torch.FloatTensor([input]).to(device)
    return input


def get_state_action_shape_from_env(env):
    # state
    if isinstance(env.observation_space, Discrete):
        state_shape = [env.observation_space.n]
    elif isinstance(env.observation_space, Box):
        state_shape = env.observation_space.shape[0]

    # action
    if isinstance(env.action_space, Discrete):
        action_shape = env.action_space.n
    elif isinstance(env.action_space, Box):
        action_shape = env.action_space.shape[0]

    return state_shape, action_shape


class WeightDecay:
    def __init__(self, start, end, decay):
        self.val = start
        self.end = end
        self.decay = decay

    def step(self):
        self.val = max(self.end, self.decay*self.val)
        return self.val


class ExperienceReplay:
    def __init__(self, num_experience=128, num_recall=64):
        self.num_experience = num_experience
        self.num_recall = num_recall
        self.memories = []
        self.position = 0

    def remember(self, *args):
        if len(self.memories) == 0:
            # init
            for i in range(len(args)):
                self.memories.append(torch.tensor(
                    [args[i]], dtype=torch.float, device=device))
        else:
            if len(self.memories[0]) < self.num_experience:
                # push
                for i in range(len(args)):
                    self.memories[i] = torch.cat(
                        (self.memories[i], torch.tensor([args[i]], dtype=torch.float, device=device)), dim=0)
            else:  # set
                for i in range(len(args)):
                    self.memories[i][self.position] = torch.tensor(
                        [args[i]], dtype=torch.float, device=device)
        self.position = (self.position + 1) % self.num_experience

    def recall(self):
        memory_length = len(self.memories[0])
        size = self.num_recall if self.num_recall < memory_length else memory_length
        query = random.sample(range(memory_length), size)
        return list([self.memories[i][query] for i in range(len(self.memories))])
