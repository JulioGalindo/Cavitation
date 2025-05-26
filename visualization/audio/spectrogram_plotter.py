import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import scipy.signal as signal
from visualization.base import BasePlotter

class SpectrogramPlotter(BasePlotter):
    """
    Plot the spectrogram (STFT) of a WAV file.
    """
    def __init__(self, nperseg: int = 1024, noverlap: int = None, cmap: str = 'viridis'):
        self.nperseg  = nperseg
        self.noverlap = noverlap or nperseg // 2
        self.cmap     = cmap

    def plot(self, wav_path: str, ax=None, **plot_kwargs):
        data, sr = sf.read(wav_path)
        f, t, Sxx = signal.spectrogram(
            data, fs=sr, nperseg=self.nperseg, noverlap=self.noverlap
        )

        if ax is None:
            fig, ax = plt.subplots()
            
        kwargs = plot_kwargs.copy()
        title  = kwargs.pop('title', 'Spectrogram')
        cmap   = kwargs.pop('cmap', self.cmap)
        cb_label = kwargs.pop('colorbar_label', 'Intensity [dB]')

        im = ax.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10),
                           shading='gouraud', cmap=cmap, **kwargs)
        ax.set_title(title)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Frequency [Hz]')

        fig = ax.get_figure()
        fig.colorbar(im, ax=ax, label=cb_label)
        return ax
