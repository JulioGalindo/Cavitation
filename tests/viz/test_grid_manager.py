import numpy as np
import matplotlib.pyplot as plt
from visualization.grid_managers import GridManager
from visualization.audio.wav_plotter import WaveformPlotter
from visualization.audio.spectrum_plotter import SpectrumPlotter
from visualization.vibration.accel_plotter import AccelPlotter


def test_grid_manager_basic(tmp_path):
    # Prepare dummy WAV
    wav_file = tmp_path / "dummy.wav"
    sr = 8000
    data = np.zeros(int(sr * 0.1), dtype=np.float32)
    import soundfile as sf
    sf.write(str(wav_file), data, sr)

    vib = np.ones(100)

    configs = [
        {'plotter': WaveformPlotter(), 'title': 'Wav', 'plot_kwargs': {}, 'source_data': None},
        {'plotter': SpectrumPlotter(), 'title': 'PSD', 'plot_kwargs': {}, 'source_data': None},
        {'plotter': AccelPlotter(sample_rate=100), 'title': 'VibTime', 'plot_kwargs': {}, 'source_data': vib},
    ]
    mgr = GridManager(wav_path=str(wav_file))
    fig, axes = mgr.plot_grid(configs, nrows=1, ncols=3, fig_title="Test")
    assert len(axes) == 3
    assert fig._suptitle.get_text() == "Test"
    plt.close(fig)
