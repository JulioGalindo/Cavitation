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

# Shared config
cfg = {
    'sample_rate': 8000,
    'duration_seconds': 0.1,
    'snr_db': 20,
    'add_pink': False,
    'burst_count_mul': 5,
    'burst_width_range': [0.0005, 0.001],
    'burst_amp_range': [0.3, 0.8],
    'envelope_type': 'gaussian',
    'rotor_freq': 100,
    'number_harmonics': 2,
    'tone_amplitudes': [0.05, 0.03],
    'mod_index': 0.2,
    'rho': 998, 'sigma': 0.072, 'mu': 0.001,
    'P_v': 2300, 'scale_factor': 0.5,
    'max_state': 6,
    'latent_dim': 16,
    'timesteps': 10,
}

gens = [
    ("Rule‐Based", RuleBasedGenerator(cfg)),
    ("Physical",   PhysicalModelGenerator(cfg)),
    ("CGAN",       CGANGenerator(cfg)),
    ("CVAE",       CVAEGenerator(cfg)),
    ("Diffusion",  DiffusionGenerator(cfg)),
]

state = 3
sr = cfg['sample_rate']
t = np.linspace(0, cfg['duration_seconds'], int(sr*cfg['duration_seconds']), endpoint=False)

fig, axes = plt.subplots(len(gens), 4, figsize=(12, 10), 
                         gridspec_kw={'wspace':0.4, 'hspace':0.6})

for row, (name, gen) in enumerate(gens):
    ts, _ = gen.generate(state)
    data = ts.data if hasattr(ts, 'data') else np.array(ts)
    
    # 1) Time waveform
    ax = axes[row, 0]
    ax.plot(t, data, lw=1)
    ax.set_title(f"{name}\nWaveform")
    ax.set_ylabel("Amp")
    if row == len(gens)-1:
        ax.set_xlabel("Time [s]")
    
    # 2) Mel spectrogram
    ax = axes[row, 1]
    S = librosa.feature.melspectrogram(y=data, sr=sr, n_fft=256, 
                                       hop_length=128, n_mels=64)
    S_db = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_db, sr=sr, hop_length=128,
                                   x_axis='time', y_axis='mel', ax=ax,
                                   cmap='magma')
    ax.set_title("Mel Spec")
    if row == len(gens)-1:
        ax.set_xlabel("Time [s]")
    
    # 3) Spectrogram (STFT)
    ax = axes[row, 2]
    f, tt, Sxx = signal.spectrogram(data, fs=sr, nperseg=128, noverlap=64)
    pcm = ax.pcolormesh(tt, f, 10*np.log10(Sxx+1e-10), shading='gouraud')
    ax.set_title("Spectrogram")
    if row == len(gens)-1:
        ax.set_xlabel("Time [s]")
    
    # 4) PSD (Welch)
    ax = axes[row, 3]
    freqs, psd = signal.welch(data, sr, nperseg=256, noverlap=128)
    ax.semilogy(freqs, psd)
    ax.set_title("PSD")
    if row == len(gens)-1:
        ax.set_xlabel("Freq [Hz]")

# Global figure title
fig.suptitle(f"Generator outputs (state={state})", fontsize=16, y=0.95)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.show()
