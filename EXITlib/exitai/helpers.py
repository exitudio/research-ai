from torch.optim.optimizer import Optimizer


def set_lr(optimizer, lr)->None:
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
