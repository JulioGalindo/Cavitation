import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from visualization.base import BasePlotter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

class Waveform3DPlotter(BasePlotter):
    """
    Plot a 3D representation of the WAV file waveform.

    X axis: time [s]
    Y axis: zero-plane (for visualization)
    Z axis: amplitude
    """
    def plot(self, wav_path: str, ax=None, **plot_kwargs):
        data, sr = sf.read(wav_path)
        times = np.arange(len(data)) / sr

        kwargs = plot_kwargs.copy()
        title = kwargs.pop('title', '3D Waveform')

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

        zero_plane = np.zeros_like(times)
        ax.plot(times, zero_plane, data, **kwargs)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('')
        ax.set_zlabel('Amplitude')
        ax.set_title(title)
        return ax
