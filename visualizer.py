import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


class Visualizer:
    """
    Lightweight visualizer for live training feedback.
    Runs on CPU to minimize GPU usage. Supports delayed plotting.
    """
    def __init__(self):
        self.epochs = []
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

        plt.ion()
        self.fig, self.axs = plt.subplots(2, 1, figsize=(10, 6))
        self.axs[0].set_title("Loss")
        self.axs[0].set_xlabel("Epoch")
        self.axs[0].set_ylabel("Loss")
        self.axs[1].set_title("Accuracy")
        self.axs[1].set_xlabel("Epoch")
        self.axs[1].set_ylabel("Accuracy")
        self.axs[0].yaxis.set_major_locator(mticker.MaxNLocator(prune='lower'))
        self.axs[1].yaxis.set_major_locator(mticker.MaxNLocator(prune='lower'))
        self.fig.tight_layout()

    def update(self, epoch, train_loss=None, val_loss=None, train_acc=None, val_acc=None):
        if epoch not in self.epochs:
            self.epochs.append(epoch)
        if train_loss is not None:
            self.train_losses.append(train_loss)
        if val_loss is not None:
            self.val_losses.append(val_loss)
        if train_acc is not None:
            self.train_accuracies.append(train_acc)
        if val_acc is not None:
            self.val_accuracies.append(val_acc)

        self._draw()

    def _draw(self):
        self.axs[0].cla()
        self.axs[1].cla()

        self.axs[0].set_title("Loss")
        self.axs[0].set_xlabel("Epoch")
        self.axs[0].set_ylabel("Loss")
        self.axs[0].plot(self.epochs[:len(self.train_losses)], self.train_losses, label="Train")
        if self.val_losses:
            self.axs[0].plot(self.epochs[:len(self.val_losses)], self.val_losses, label="Val")
        self.axs[0].legend()

        self.axs[1].set_title("Accuracy")
        self.axs[1].set_xlabel("Epoch")
        self.axs[1].set_ylabel("Accuracy")
        self.axs[1].plot(self.epochs[:len(self.train_accuracies)], self.train_accuracies, label="Train")
        if self.val_accuracies:
            self.axs[1].plot(self.epochs[:len(self.val_accuracies)], self.val_accuracies, label="Val")
        self.axs[1].legend()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def finalize(self):
        plt.ioff()
        plt.show()
