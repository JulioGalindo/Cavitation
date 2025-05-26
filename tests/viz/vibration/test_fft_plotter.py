# tests/viz/vibration/test_fft_plotter.py
import numpy as np
import matplotlib.pyplot as plt
import pytest
from visualization.vibration.fft_plotter import FFTPlotter

@pytest.fixture
def sine_data():
    sr = 1000
    t = np.linspace(0, 1, sr, endpoint=False)
    data = np.sin(2 * np.pi * 50 * t)  # 50 Hz sine
    return data, sr

def test_fft_plotter(sine_data):
    data, sr = sine_data
    plotter = FFTPlotter(sample_rate=sr)
    fig, ax = plt.subplots()
    ax_out = plotter.plot(data, ax=ax, color='r', title='FFT Test', ylabel='Amp')
    assert ax_out is ax
    assert ax.get_title() == 'FFT Test'
    assert ax.get_xlabel() == 'Frequency [Hz]'
    assert ax.get_ylabel() == 'Amp'

    lines = ax.get_lines()
    assert len(lines) == 1
    freqs, mags = lines[0].get_data()
    # frequencies ascend and magnitudes non-negative
    assert np.all(freqs[1:] > freqs[:-1])
    assert np.all(mags >= 0)
    # check peak near 50 Hz
    peak_idx = np.argmax(mags)
    assert freqs[peak_idx] == pytest.approx(50, abs=1)
    plt.close(fig)
