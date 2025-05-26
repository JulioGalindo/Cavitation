import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import scipy.signal as signal
import librosa
import librosa.display

from generators.rule_based import RuleBasedGenerator
from generators.physical import PhysicalModelGenerator
from generators.cgan import CGANGenerator
from generators.cvae import CVAEGenerator
from generators.diffusion import DiffusionGenerator

# Configuración común
cfg = {
    'sample_rate': 8000,
    'duration_seconds': 0.1,
    # …tus otros params…
}

gens = [
    ("Rule-Based", RuleBasedGenerator(cfg)),
    ("Physical",   PhysicalModelGenerator(cfg)),
    ("CGAN",       CGANGenerator(cfg)),
    ("CVAE",       CVAEGenerator(cfg)),
    ("Diffusion",  DiffusionGenerator(cfg)),
]

state = 3
sr = cfg['sample_rate']
t = np.linspace(0, cfg['duration_seconds'], int(sr*cfg['duration_seconds']), endpoint=False)

# 1) Waveforms
fig = plt.figure("Waveforms")
for i, (name, gen) in enumerate(gens, start=1):
    ts, _ = gen.generate(state)
    data = getattr(ts, 'data', np.array(ts))
    ax = fig.add_subplot(len(gens), 1, i)
    ax.plot(t, data, lw=1)
    ax.set_ylabel(name)
    if i == len(gens):
        ax.set_xlabel("Time [s]")
fig.tight_layout()

# 2) Mel Spectrograms
fig = plt.figure("Mel Spectrograms")
for i, (_, gen) in enumerate(gens, start=1):
    ts, _ = gen.generate(state)
    data = getattr(ts, 'data', np.array(ts))
    ax = fig.add_subplot(len(gens), 1, i)
    S = librosa.feature.melspectrogram(y=data, sr=sr, n_fft=256, hop_length=128, n_mels=64)
    S_db = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_db, sr=sr, hop_length=128,
                             x_axis=None, y_axis='mel', ax=ax, cmap='magma')
    if i == len(gens):
        ax.set_xlabel("Time [s]")
fig.tight_layout()

# 3) STFT Spectrograms
fig = plt.figure("STFT Spectrograms")
for i, (_, gen) in enumerate(gens, start=1):
    ts, _ = gen.generate(state)
    data = getattr(ts, 'data', np.array(ts))
    ax = fig.add_subplot(len(gens), 1, i)
    f, tt, Sxx = signal.spectrogram(data, fs=sr, nperseg=128, noverlap=64)
    ax.pcolormesh(tt, f, 10*np.log10(Sxx+1e-10), shading='gouraud')
    if i == len(gens):
        ax.set_xlabel("Time [s]")
fig.tight_layout()

# 4) PSDs
fig = plt.figure("PSDs")
for i, (_, gen) in enumerate(gens, start=1):
    ts, _ = gen.generate(state)
    data = getattr(ts, 'data', np.array(ts))
    ax = fig.add_subplot(len(gens), 1, i)
    freqs, psd = signal.welch(data, sr, nperseg=256, noverlap=128)
    ax.semilogy(freqs, psd)
    if i == len(gens):
        ax.set_xlabel("Frequency [Hz]")
fig.tight_layout()

plt.show()
