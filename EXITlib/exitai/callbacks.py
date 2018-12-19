from .helpers import set_lr
import matplotlib.pyplot as plt
import torch
import numpy as np


class Callback:
    def is_done(self)->bool: pass

    def on_train_begin(self): pass

    def on_epoch_begin(self): pass

    def on_batch_begin(self): pass

    def on_batch_end(self, loss, output, target): pass

    def on_epoch_end(self, phase, num_data): pass

    def on_train_end(self): pass


class MultipleCallbacks(Callback):
    def __init__(self, *callbacks):
        self.callbacks = callbacks

        # get all methods name from Callback except is_done
        function_names = (func for func in dir(Callback)
                          if callable(getattr(Callback, func)) and
                          not func.startswith("__") and
                          func is not "is_done"
                          )

        # override all functions dynamically
        for function_name in function_names:
            def bind_function(function_name):
                def new_function(*arg):
                    for cb in self.callbacks:
                        getattr(cb, function_name)(*arg)
                return new_function
            setattr(self, function_name, bind_function(function_name))

    def is_done(self)->bool:
        for cb in self.callbacks:
            if cb.is_done():
                return True


class FitTracker(Callback):
    def __init__(self):
        super().__init__()
        self.total_loss = 0
        self.total_correct = 0

    def on_batch_end(self, loss, output, target):
        # ------ find correct ----------
        # max of dimension 1, keepdim, and [0] is value / [1] is index (we need only index)
        pred = output.max(1)[1]  # get the index of the max log-probability
        correct = pred.eq(target).sum().item()
        # ------------------------------
        self.total_loss += loss.item()
        self.total_correct += correct

    def on_epoch_end(self, phase, num_data):
        epoch_loss = self.total_loss / num_data*100
        epoch_acc = self.total_correct / num_data*100
        print('   [{}] Average loss: {:.4f}, acc: {:.2f}%'.format(
            phase, epoch_loss, epoch_acc))
        self.total_loss = 0
        self.total_correct = 0


class CosineAnnealingLR_Updater(Callback):
    def __init__(self, optimizer, cycle_num_batch, cycle_len, cycle_mult):
        self.optimizer = optimizer
        self.cycle_num_batch = cycle_num_batch
        self.cycle_len = cycle_len
        self.cycle_mult = cycle_mult

    def _update_scheduler(self):
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, self.cycle_num_batch)

    def on_train_begin(self):
        self.cycle_iter, self.cycle_count = 0, 0
        self._update_scheduler()

    def on_batch_begin(self):
        self.scheduler.step()

    def on_batch_end(self, *args):
        self.cycle_iter += 1
        if self.cycle_iter == self.cycle_num_batch:
            self.cycle_iter = 0
            self.cycle_num_batch *= self.cycle_mult
            self.cycle_count += 1
            self._update_scheduler()


class LRTracker(Callback):
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.lrs = []

    def on_batch_begin(self):
        batch_lrs = []
        for param_group in self.optimizer.param_groups:
            batch_lrs.append(param_group['lr'])
        self.lrs.append(batch_lrs)

    def on_train_end(self):
        lrs = np.array(self.lrs)
        plt.plot(lrs[:, 0])


class LRFinder(Callback):
    def __init__(self, start_lr, end_lr, num_it, optimizer, num_batch=10):
        super().__init__()
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.num_it = num_it
        self.optimizer = optimizer
        self.num_batch = num_batch

        self.losses = []
        self.total_loss = 0
        self.lrs = []
        self.count = 0

    def annealing_exp(self, start: float, end: float, pct: float)->float:
        "Exponentially anneal from `start` to `end` as pct goes from 0.0 to 1.0."
        return start * (end/start) ** pct

    def update_lr(self):
        lr = self.annealing_exp(
            self.start_lr, self.end_lr, len(self.lrs)/self.num_it)
        set_lr(self.optimizer, lr)
        return lr

    def on_batch_end(self, loss, output, target):
        self.total_loss += loss.item()
        self.count += 1
        if self.count == self.num_batch:
            self.lrs.append(self.update_lr())
            self.losses.append(self.total_loss/self.num_batch)
            self.count = 0
            self.total_loss = 0

    def is_done(self): return len(self.losses) >= self.num_it
