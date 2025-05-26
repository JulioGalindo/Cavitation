import numpy as np
import matplotlib.pyplot as plt
from visualization.base import BasePlotter

class AccelPlotter(BasePlotter):
    """
    Plot time-domain accelerometer data.

    X axis: time [s]
    Y axis: acceleration [unit]
    """
    def __init__(self, sample_rate: float):
        self.sample_rate = sample_rate

    def plot(self, data: np.ndarray, ax=None, **plot_kwargs):
        times = np.arange(len(data)) / self.sample_rate

        kwargs = plot_kwargs.copy()
        title  = kwargs.pop('title', 'Acceleration vs Time')
        ylabel = kwargs.pop('ylabel', 'Acceleration')

        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(times, data, **kwargs)
        ax.set_title(title)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel(ylabel)
        return ax
