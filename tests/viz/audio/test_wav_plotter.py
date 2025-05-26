import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import pytest
from visualization.audio.wav_plotter import WaveformPlotter, RMSPlotter

def generate_sine_wav(path, freq=440.0, sr=8000, duration=0.1):
    t = np.linspace(0, duration, int(sr*duration), endpoint=False)
    data = 0.5 * np.sin(2 * np.pi * freq * t)
    sf.write(path, data.astype(np.float32), sr)
    return data, sr

@pytest.fixture
def wav_file(tmp_path):
    wav_path = tmp_path / 'sine.wav'
    data, sr = generate_sine_wav(str(wav_path))
    return str(wav_path), data, sr

def test_waveform_plotter(wav_file):
    wav_path, data, sr = wav_file
    plotter = WaveformPlotter()
    fig, ax = plt.subplots()
    ax_out = plotter.plot(wav_path, ax=ax, color='r', title='Test Waveform')
    assert ax_out is ax
    assert ax.get_title() == 'Test Waveform'
    assert ax.get_xlabel() == 'Time [s]'
    assert ax.get_ylabel() == 'Amplitude'

    lines = ax.get_lines()
    assert len(lines) == 1
    x_data, y_data = lines[0].get_data()

    # Compare con tolerancia más laxa
    for yd, true_d in zip(y_data, data):
        assert yd == pytest.approx(true_d, rel=1e-3, abs=1e-3)
    expected_times = np.arange(len(data)) / sr
    for xt, true_t in zip(x_data, expected_times):
        assert xt == pytest.approx(true_t, rel=1e-6)

    plt.close(fig)

def test_rms_plotter(wav_file):
    wav_path, data, sr = wav_file
    frame_size = len(data) // 2
    hop_size = frame_size
    plotter = RMSPlotter(frame_size=frame_size, hop_size=hop_size)
    fig, ax = plt.subplots()
    ax_out = plotter.plot(wav_path, ax=ax, color='g', title='Test RMS')
    assert ax_out is ax

    expected_rms = np.sqrt(np.mean(data[:frame_size]**2))
    lines = ax.get_lines()
    assert len(lines) == 1
    y = lines[0].get_ydata()
    # Debe haber dos frames
    assert y.shape[0] == 2
    # Ambas envolventes iguales para seno puro
    assert y[0] == pytest.approx(expected_rms, rel=1e-3, abs=1e-3)
    assert y[1] == pytest.approx(expected_rms, rel=1e-3, abs=1e-3)

    plt.close(fig)
