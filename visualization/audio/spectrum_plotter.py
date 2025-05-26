import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import scipy.signal as signal
from visualization.base import BasePlotter

class SpectrumPlotter(BasePlotter):
    """
    Plot the Power Spectral Density (PSD) of a WAV file using Welch's method.
    """
    def __init__(self, nperseg: int = 1024, noverlap: int = None):
        self.nperseg = nperseg
        self.noverlap = noverlap or nperseg // 2

    def plot(self, wav_path: str, ax=None, **plot_kwargs):
        data, sr = sf.read(wav_path)
        freqs, psd = signal.welch(data, sr, nperseg=self.nperseg, noverlap=self.noverlap)

        if ax is None:
            fig, ax = plt.subplots()

        kwargs = plot_kwargs.copy()
        title = kwargs.pop('title', 'Power Spectral Density')

        ax.semilogy(freqs, psd, **kwargs)
        ax.set_title(title)
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('PSD')
        return ax
