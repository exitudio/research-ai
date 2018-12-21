import torch
import matplotlib.pyplot as plt
from typing import Callable
from .callbacks import FitTracker, LRTracker, LRFinder, MultipleCallbacks, Callback, CosineAnnealingLR_Updater, FitTrackerWithSaveAndEarlyStopping, classification_eval_func
from .helpers import get_param_groups_with_lr, get_callbacks_by_tuple_string


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
                *input_data, target = (i.to(self.device) for i in data)
                output = self.model(*input_data) 

                loss = self.criterion(output, target)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # print('target:', target)
                callback.on_batch_end(loss, output, target)
                if callback.is_done():
                    break

        callback.on_epoch_end(phase, len(data_loader.dataset))

    def fit(self, lr: float=0.003, num_epochs: int=25, cycle: dict = {'cycle_len': 2, 'cycle_mult': 2}, callbacks: any=None, eval_func=classification_eval_func)->None:
        model_params = get_param_groups_with_lr(self.model, lr)
        optimizer = torch.optim.Adam(model_params)
#         optimizer = torch.optim.SGD(model_params, momentum=0.9, weight_decay=5e-4)

        # train callback
        if cycle is not None:
            train_callback = MultipleCallbacks(
                CosineAnnealingLR_Updater(optimizer,
                                          len(self.data_loader_train) *
                                          cycle['cycle_len'],
                                          cycle['cycle_len'],
                                          cycle['cycle_mult']),
                FitTracker(eval_func)
            )
        else:
            train_callback = MultipleCallbacks(FitTracker(eval_func))

        if callbacks:
            train_callback.append(
                get_callbacks_by_tuple_string(callbacks, optimizer))

        # test callback
        test_callback = FitTrackerWithSaveAndEarlyStopping(eval_func, self.model)

        # start fit
        train_callback.on_train_begin()
        test_callback.on_train_begin()
        for i in range(num_epochs):
            print(f'---- epoch:{i} ------')
            self._run_model('train', optimizer, train_callback)
            self._run_model('test', optimizer, test_callback)
            if test_callback.is_done(): break
        train_callback.on_train_end()
        test_callback.on_train_end()

    def predict(self, eval_func=classification_eval_func):
        self._run_model('test', None, FitTracker(eval_func))

    def lr_find(self, start_lr: float=1e-7, end_lr: float=10, num_it: int=100, num_batch: int=10)->None:
        optimizer = torch.optim.Adam(self.model.parameters(), lr=start_lr)
        lr_finder = LRFinder(start_lr, end_lr, num_it,
                             optimizer, num_batch=num_batch)
        while True:
            self._run_model('train', optimizer, lr_finder)
            if lr_finder.is_done():
                break
        plt.semilogx(lr_finder.lrs, lr_finder.losses)
