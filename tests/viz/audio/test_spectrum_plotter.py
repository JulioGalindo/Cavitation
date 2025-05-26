import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import pytest
from visualization.audio.spectrum_plotter import SpectrumPlotter

# Generate test WAV
@pytest.fixture
def wav_file(tmp_path):
    sr = 8000
    t = np.linspace(0, 0.1, int(sr * 0.1), endpoint=False)
    # sum of two sine waves
    data = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.25 * np.sin(2 * np.pi * 880 * t)
    wav_path = tmp_path / 'test.wav'
    sf.write(str(wav_path), data.astype(np.float32), sr)
    return str(wav_path), data, sr


def test_spectrum_plotter(wav_file):
    wav_path, data, sr = wav_file
    plotter = SpectrumPlotter(nperseg=256, noverlap=128)
    fig, ax = plt.subplots()
    ax_out = plotter.plot(wav_path, ax=ax, color='b', title='Test PSD')
    assert ax_out is ax
    # Check axes labels
    assert ax.get_xlabel() == 'Frequency [Hz]'
    assert ax.get_ylabel() == 'PSD'
    assert ax.get_title() == 'Test PSD'
    # Check that line is plotted
    lines = ax.get_lines()
    assert len(lines) == 1
    freqs, psd = lines[0].get_data()
    # frequencies should be increasing
    assert np.all(freqs[1:] > freqs[:-1])
    # PSD values non-negative
    assert np.all(psd >= 0)
    plt.close(fig)
