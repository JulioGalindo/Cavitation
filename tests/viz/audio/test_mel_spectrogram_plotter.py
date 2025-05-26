import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import pytest
from matplotlib.image import AxesImage
from matplotlib.collections import QuadMesh
from visualization.audio.mel_spectrogram_plotter import MelSpectrogramPlotter

def generate_sine_wav(path, freq=440.0, sr=8000, duration=0.1):
    t = np.linspace(0, duration, int(sr*duration), endpoint=False)
    data = 0.5 * np.sin(2 * np.pi * freq * t)
    sf.write(path, data.astype(np.float32), sr)
    return data, sr

@pytest.fixture
def wav_file(tmp_path):
    wav_path = tmp_path / 'test.wav'
    data, sr = generate_sine_wav(str(wav_path))
    return str(wav_path), data, sr

def test_mel_spectrogram_plotter(wav_file):
    wav_path, _, sr = wav_file
    plotter = MelSpectrogramPlotter(n_fft=256, hop_length=128, n_mels=64)
    fig, ax = plt.subplots()
    ax_out = plotter.plot(
        wav_path, ax=ax,
        title='Mel Spec', cmap='magma',
        colorbar_label='dB Scale'
    )
    assert ax_out is ax
    assert ax.get_title() == 'Mel Spec'
    assert ax.get_xlabel() == 'Time [s]'
    assert ax.get_ylabel() == 'Mel Frequency'

    # There should be at least one AxesImage or QuadMesh
    artists = ax.get_children()
    imgs = [c for c in artists if isinstance(c, (AxesImage, QuadMesh))]
    assert len(imgs) >= 1

    # Verify the colorbar label
    cbar = fig.axes[-1]
    assert 'dB Scale' in cbar.get_ylabel()

    plt.close(fig)
