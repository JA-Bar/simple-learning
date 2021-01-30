from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from .tensor import Tensor


class TrainLogger:
    def __init__(self):
        self.train_loss_log = []
        self.val_loss_log = []
        self.log_step = 0

        self.train_metric_log = defaultdict(list)
        self.val_metric_log = defaultdict(list)
        self.metric_step = 0

    def log_loss(self, loss, mode='train'):
        if not isinstance(loss, (int, float)):
            loss = loss.item()

        if 'train' in mode:
            self.train_loss_log.append((self.log_step, loss))
            self.log_step += 1
        elif 'val' in mode:
            self.val_loss_log.append((self.log_step, loss))
        else:
            raise ValueError("Mode not compatible, select either train or val")

    def log_metrics(self, predictions, labels, mode='train', metric='accuracy'):
        if isinstance(predictions, Tensor):
            predictions = predictions.data

        if isinstance(labels, Tensor):
            labels = labels.data

        if 'acc' in metric:
            axis = len(predictions.shape) - 1
            n = labels.shape[0]

            predicted_labels = np.argmax(predictions, axis=axis)
            correct = (predicted_labels == labels)
            metric_result = np.sum(correct) / n

        if 'train' in mode:
            self.train_metric_log[metric].append((self.metric_step, metric_result))
            self.metric_step += 1
        elif 'val' in mode:
            self.val_metric_log[metric].append((self.metric_step, metric_result))
        else:
            raise ValueError("Mode not compatible, select either train or val")

    def default_plotting_settings(self, figsize=(7, 7), dpi=100):
        plt.style.use(['dark_background', 'bmh'])
        plt.rc('axes', facecolor='k')
        plt.rc('figure', facecolor='k')
        plt.rc('figure', figsize=figsize, dpi=dpi)

    def plot_loss(self, train=True, val=True):
        self.default_plotting_settings()

        if train and self.train_loss_log:
            plt.plot(*list(zip(*self.train_loss_log)), label='Train')
        if val and self.val_loss_log:
            plt.plot(*list(zip(*self.val_loss_log)), label='Validation')

        plt.title('Loss during training')
        plt.xlabel('Training steps')
        plt.ylabel('Loss')

        plt.legend()
        plt.show()

    def plot_metrics(self, train=True, val=True):
        self.default_plotting_settings()

        if train:
            for metric, values in self.train_metric_log.items():
                plt.plot(*list(zip(*values)), label=f'Train {metric}')
        if val:
            for metric, values in self.val_metric_log.items():
                plt.plot(*list(zip(*values)), label=f'Validation {metric}')

        plt.title('Metrics during training')
        plt.xlabel('Training steps')
        plt.ylabel('Metrics')

        plt.legend()
        plt.show()


class DataLoader:
    def __init__(self, data, targets, batch_size):
        self.data = data
        self.targets = targets
        self.batch_size = batch_size

        self.current_index = 0
        self.dataset_size = data.shape[0]

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_index >= self.dataset_size:
            self.current_index = 0
            raise StopIteration

        if self.current_index // self.dataset_size == 0:
            num_samples = self.batch_size
        else:
            num_samples = self.batch_size - (self.dataset_size % self.current_index)

        data = self.data[self.current_index:self.current_index+num_samples]
        target = self.targets[self.current_index:self.current_index+num_samples]

        self.current_index += self.batch_size

        return (data, target)

    def __len__(self):
        return self.dataset_size

