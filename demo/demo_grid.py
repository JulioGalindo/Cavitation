import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from visualization.grid_managers import GridManager
from visualization.audio.wav_plotter import WaveformPlotter
from visualization.audio.spectrum_plotter import SpectrumPlotter
from visualization.vibration.accel_plotter import AccelPlotter

# 1) Generar un WAV de ejemplo
sr_audio = 8000
duration = 0.1
t_audio = np.linspace(0, duration, int(sr_audio * duration), endpoint=False)
audio_data = np.sin(2 * np.pi * 440 * t_audio).astype(np.float32)
audio_path = "demo.wav"
sf.write(audio_path, audio_data, sr_audio)

# 2) Generar datos de vibración de ejemplo
sr_vib = 2000
t_vib = np.linspace(0, 1, sr_vib, endpoint=False)
vib_data = 0.1 * np.sin(2 * np.pi * 30 * t_vib)

# 3) Configurar los plotters
configs = [
    {
        "plotter": WaveformPlotter(),
        "title": "Audio Waveform",
        "description": "440 Hz tone",
        "plot_kwargs": {"color": "blue"},
        "source_data": None,
    },
    {
        "plotter": SpectrumPlotter(),
        "title": "Audio PSD",
        "plot_kwargs": {"color": "green"},
        "source_data": None,
    },
    {
        "plotter": AccelPlotter(sample_rate=sr_vib),
        "title": "Vibration Time",
        "description": "30 Hz accel",
        "plot_kwargs": {"color": "orange", "ylabel": "g"},
        "source_data": vib_data,
    },
]

# 4) Crear el GridManager y mostrar la cuadrícula 1×3
mgr = GridManager(wav_path=audio_path)
fig, axes = mgr.plot_grid(
    configs, nrows=1, ncols=3, fig_title="Audio & Vibration Demo"
)
plt.show()
