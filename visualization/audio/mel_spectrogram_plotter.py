import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from visualization.base import BasePlotter
import librosa
import librosa.display

class MelSpectrogramPlotter(BasePlotter):
    """
    Plot the Mel-scaled spectrogram of a WAV file.
    """
    def __init__(self, n_fft: int = 1024, hop_length: int = 512, n_mels: int = 128):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

    def plot(self, wav_path: str, ax=None, **plot_kwargs):
        data, sr = sf.read(wav_path)
        # compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=data, sr=sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)

        if ax is None:
            fig, ax = plt.subplots()

        # extract title and styling kwargs
        kwargs = plot_kwargs.copy()
        title = kwargs.pop('title', 'Mel Spectrogram')
        cmap = kwargs.pop('cmap', 'viridis')
        cbar_label = kwargs.pop('colorbar_label', 'dB')

        img = librosa.display.specshow(
            mel_db, sr=sr, hop_length=self.hop_length,
            x_axis='time', y_axis='mel', ax=ax,
            cmap=cmap, **kwargs
        )
        ax.set_title(title)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Mel Frequency')

        fig = ax.get_figure()
        fig.colorbar(img, ax=ax, label=cbar_label)
        return ax
