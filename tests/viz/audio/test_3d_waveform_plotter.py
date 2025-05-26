import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import pytest
from mpl_toolkits.mplot3d.art3d import Line3D
from visualization.audio.waveform_3d_plotter import Waveform3DPlotter

@pytest.fixture
def wav_file(tmp_path):
    sr = 8000
    t = np.linspace(0, 0.1, int(sr * 0.1), endpoint=False)
    data = 0.5 * np.sin(2 * np.pi * 440 * t)
    wav_path = tmp_path / '3d.wav'
    sf.write(str(wav_path), data.astype(np.float32), sr)
    return str(wav_path), data, sr

def test_waveform_3d_plotter(wav_file):
    wav_path, _, sr = wav_file
    plotter = Waveform3DPlotter()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax_out = plotter.plot(wav_path, ax=ax, color='c', title='3D Test')
    assert ax_out is ax
    assert ax.get_title() == '3D Test'
    assert ax.get_xlabel() == 'Time [s]'
    assert ax.get_ylabel() == ''
    assert ax.get_zlabel() == 'Amplitude'

    lines = [c for c in ax.get_children() if isinstance(c, Line3D)]
    assert len(lines) == 1

    plt.close(fig)
