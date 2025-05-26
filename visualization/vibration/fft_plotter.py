import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from visualization.base import BasePlotter

class FFTPlotter(BasePlotter):
    """
    Plot magnitude FFT of vibration data.

    X axis: frequency [Hz]
    Y axis: magnitude
    """
    def __init__(self, sample_rate: float):
        self.sample_rate = sample_rate

    def plot(self, data: np.ndarray, ax=None, **plot_kwargs):
        N = len(data)
        fft_vals = np.abs(rfft(data))
        freqs   = rfftfreq(N, 1/self.sample_rate)

        kwargs = plot_kwargs.copy()
        title  = kwargs.pop('title', 'FFT Magnitude')
        ylabel = kwargs.pop('ylabel', 'Magnitude')

        if ax is None:
            fig, ax = plt.subplots()

        ax.semilogy(freqs, fft_vals, **kwargs)
        ax.set_title(title)
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel(ylabel)
        return ax
