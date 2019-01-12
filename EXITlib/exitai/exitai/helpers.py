from torch.optim.optimizer import Optimizer
import math
from .const import LR_TRACKER
import exitai.callbacks


def set_lr(optimizer, lr)->None:
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_value_in_log_scale(vals, current_index, length):
    # avoid divide by zero
    if length == 1:
        return vals[len(vals)-1]

    index_ratio = current_index/(length-1)
    min_index = math.floor(index_ratio/(len(vals)-1))
    max_index = math.ceil(index_ratio/(len(vals)-1))

    # if divisible
    if min_index == max_index:
        return vals[min_index]

    min_log = math.log(vals[min_index])
    max_log = math.log(vals[max_index])

    new_val_linear_scale = current_index/(length-1)
    new_val_log_scale = new_val_linear_scale * (max_log - min_log)
    val = math.pow(math.e, min_log + new_val_log_scale)
    return round(val, 4)


def get_param_groups_with_lr(model, lrs):
    num_children = len([c for c in model.children()])
    param_groups = []
    for i, child in enumerate(model.children()):
        if type(lrs) is list:
            lr = get_value_in_log_scale(lrs, i, num_children)
        else:
            lr = lrs
        param_groups.append({
            'params': child.parameters(), 'lr': lr
        })
    return param_groups


def get_callbacks_by_tuple(callbacks_input, optimizer):
    # TODO maybe there is a better structure, cause this needs optimzer to initiate LRTracker
    if type(callbacks_input) is not list:
        callbacks_input = [callbacks_input]
    callbacks = []
    for callback in callbacks_input:
        if callback is LR_TRACKER:
            callbacks.append(exitai.callbacks.LRTracker(optimizer)) # TODO findout how to avoid circular import
        else:
            callbacks.append(callback)
    return callbacks
