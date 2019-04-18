from collections import deque
import random
import torch
import numpy as np


def print_weight_size(model):
    for name, param in model.named_parameters():
        if param.requires_grad and "weight" in name:
            print(name, param.data.shape)


def copy_params(from_model, to_model):
    for target_param, param in zip(to_model.parameters(), from_model.parameters()):
        target_param.data.copy_(param.data)


def convert_to_tensor(input):
    if type(input) is np.ndarray:
        input = torch.from_numpy(input).float()
    else:
        input = torch.FloatTensor([input])
    return input


class ExperienceReplay:
    def __init__(self, num_experience=128):
        self.num_experience = num_experience
        self.memories = []

    def remember(self, *args):
        if len(self.memories) == 0:
            for i in range(len(args)):
                self.memories.append(torch.Tensor([args[i]]))
        else:
            for i in range(len(args)):
                self.memories[i] = torch.cat(
                    (self.memories[i], torch.Tensor([args[i]])), dim=0)

    def recall(self, batch_size):
        size = batch_size if batch_size < len(
            self.memories) else len(self.memories[0])
        query = random.sample(range(len(self.memories[0])), size)
        return list([self.memories[i][query] for i in range(len(self.memories))])
