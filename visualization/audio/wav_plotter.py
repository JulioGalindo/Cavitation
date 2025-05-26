import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from visualization.base import BasePlotter

class WaveformPlotter(BasePlotter):
    """
    Plot the time-domain waveform of a WAV file.
    """
    def plot(self, wav_path: str, ax=None, **plot_kwargs):
        # Read data
        data, sr = sf.read(wav_path)
        times = np.arange(len(data)) / sr
        # Extract title and other plotting kwargs
        kwargs = plot_kwargs.copy()
        title = kwargs.pop('title', 'Waveform')
        # Create axes if needed
        if ax is None:
            fig, ax = plt.subplots()
        # Plot waveform
        ax.plot(times, data, **kwargs)
        ax.set_title(title)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Amplitude')
        return ax

class RMSPlotter(BasePlotter):
    """
    Plot the RMS envelope of a WAV file over time.
    """
    def __init__(self, frame_size: int = 1024, hop_size: int = 512):
        self.frame_size = frame_size
        self.hop_size = hop_size

    def plot(self, wav_path: str, ax=None, **plot_kwargs):
        # Read data
        data, sr = sf.read(wav_path)
        length = len(data)
        # Compute number of frames
        num_frames = 1 + max(0, (length - self.frame_size) // self.hop_size)
        rms = np.zeros(num_frames)
        for i in range(num_frames):
            start = i * self.hop_size
            frame = data[start:start + self.frame_size]
            rms[i] = np.sqrt(np.mean(frame**2))
        times = (np.arange(num_frames) * self.hop_size + self.frame_size / 2) / sr
        # Extract title and other plotting kwargs
        kwargs = plot_kwargs.copy()
        title = kwargs.pop('title', 'RMS Envelope')
        # Create axes if needed
        if ax is None:
            fig, ax = plt.subplots()
        # Plot RMS envelope
        ax.plot(times, rms, **kwargs)
        ax.set_title(title)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('RMS')
        return ax
