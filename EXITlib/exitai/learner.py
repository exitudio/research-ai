import torch
import matplotlib.pyplot as plt
from typing import Callable
from .callbacks import FitTracker, LRTracker, LRFinder, MultipleCallbacks, Callback, CosineAnnealingLR_Updater


class Learner:
    def __init__(self, data_loader_train, data_loader_test, model, criterion):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.data_loader_train = data_loader_train
        self.data_loader_test = data_loader_test
        self.model = model.to(self.device)
        self.criterion = criterion

    def _run_model(self, phase, optimizer, callback: Callable=Callback()):

        callback.on_epoch_begin()
        if phase == 'train':
            self.model.train()
            data_loader = self.data_loader_train
        else:
            self.model.eval()  # to tell model to adjust to test phase (eg. no dropout)
            data_loader = self.data_loader_test

        with torch.set_grad_enabled(phase == 'train'):
            for data in data_loader:
                callback.on_batch_begin()
                data, target = (i.to(self.device) for i in data)
                output = self.model(data)
                loss = self.criterion(output, target)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                callback.on_batch_end(loss, output, target)
                if callback.is_done():
                    break

        callback.on_epoch_end(phase, len(data_loader.dataset))

    def fit(self, lr: float=0.003, num_epochs: int=25, cycle_len: int=2, cycle_mult: int=2)->None:
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # train callback
        if cycle_len is not 0:
            train_callback = MultipleCallbacks(
                CosineAnnealingLR_Updater(optimizer,
                                          len(self.data_loader_train) *
                                          cycle_len,
                                          cycle_len,
                                          cycle_mult),
                FitTracker(),
                LRTracker(optimizer)
            )
        else:
            train_callback = MultipleCallbacks(
                FitTracker(),
                LRTracker(optimizer)
            )
        # test callback
        test_callback = FitTracker()

        # start fit
        train_callback.on_train_begin()
        test_callback.on_train_begin()
        for i in range(num_epochs):
            print(f'---- epoch:{i} ------')
            self._run_model('train', optimizer, train_callback)
            self._run_model('test', optimizer, test_callback)
        train_callback.on_train_end()
        test_callback.on_train_end()

    def lr_find(self, start_lr: float=1e-7, end_lr: float=10, num_it: int=100)->None:
        optimizer = torch.optim.Adam(self.model.parameters(), lr=start_lr)
        lr_finder = LRFinder(start_lr, end_lr, num_it, optimizer)
        while True:
            self._run_model('train', optimizer, lr_finder)
            if lr_finder.is_done():
                break
        plt.semilogx(lr_finder.lrs, lr_finder.losses)
