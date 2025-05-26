import sys
import numpy as np
import scipy.signal as signal
import librosa
import librosa.display

from PyQt5 import QtWidgets                   # or `from PySide2 import QtWidgets`
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from generators.rule_based import RuleBasedGenerator
from generators.physical import PhysicalModelGenerator
from generators.cgan import CGANGenerator
from generators.cvae import CVAEGenerator
from generators.diffusion import DiffusionGenerator

class GeneratorTabs(QtWidgets.QMainWindow):
    def __init__(self, cfg, state=3):
        super().__init__()
        self.setWindowTitle("Generators Comparison")
        self.cfg = cfg
        self.state = state
        self.sr = cfg['sample_rate']
        self.t = np.linspace(0,
                             cfg['duration_seconds'],
                             int(self.sr * cfg['duration_seconds']),
                             endpoint=False)

        self.gens = [
            ("Rule-Based", RuleBasedGenerator(cfg)),
            ("Physical",   PhysicalModelGenerator(cfg)),
            ("CGAN",       CGANGenerator(cfg)),
            ("CVAE",       CVAEGenerator(cfg)),
            ("Diffusion",  DiffusionGenerator(cfg)),
        ]

        tabs = QtWidgets.QTabWidget()
        tabs.addTab(self._make_waveform_tab(), "Waveforms")
        tabs.addTab(self._make_mel_tab(),      "Mel Spec")
        tabs.addTab(self._make_stft_tab(),     "STFT Spec")
        tabs.addTab(self._make_psd_tab(),      "PSD")
        self.setCentralWidget(tabs)
        self.resize(800, 600)
        self.show()

    def _make_canvas(self):
        fig = Figure()
        return fig, fig.add_subplot(111)

    def _make_waveform_tab(self):
        fig = Figure(figsize=(6,4))
        canvas = FigureCanvas(fig)
        axes = [fig.add_subplot(len(self.gens),1,i+1)
                for i in range(len(self.gens))]
        for ax, (name, gen) in zip(axes, self.gens):
            ts,_ = gen.generate(self.state)
            data = getattr(ts,'data',np.array(ts))
            ax.plot(self.t, data, lw=1)
            ax.set_ylabel(name)
        axes[-1].set_xlabel("Time [s]")
        fig.tight_layout()
        return canvas

    def _make_mel_tab(self):
        fig = Figure(figsize=(6,4))
        canvas = FigureCanvas(fig)
        axes = [fig.add_subplot(len(self.gens),1,i+1)
                for i in range(len(self.gens))]
        for ax, (_, gen) in zip(axes, self.gens):
            ts,_ = gen.generate(self.state)
            data = getattr(ts,'data',np.array(ts))
            S = librosa.feature.melspectrogram(
                y=data, sr=self.sr,
                n_fft=256, hop_length=128, n_mels=64
            )
            S_db = librosa.power_to_db(S, ref=np.max)
            librosa.display.specshow(
                S_db, sr=self.sr, hop_length=128,
                x_axis=None, y_axis='mel',
                ax=ax, cmap='magma'
            )
        axes[-1].set_xlabel("Time [s]")
        fig.tight_layout()
        return canvas

    def _make_stft_tab(self):
        fig = Figure(figsize=(6,4))
        canvas = FigureCanvas(fig)
        axes = [fig.add_subplot(len(self.gens),1,i+1)
                for i in range(len(self.gens))]
        for ax, (_, gen) in zip(axes, self.gens):
            ts,_ = gen.generate(self.state)
            data = getattr(ts,'data',np.array(ts))
            f, tt, Sxx = signal.spectrogram(
                data, fs=self.sr, nperseg=128, noverlap=64
            )
            ax.pcolormesh(tt, f, 10*np.log10(Sxx+1e-10),
                          shading='gouraud')
        axes[-1].set_xlabel("Time [s]")
        fig.tight_layout()
        return canvas

    def _make_psd_tab(self):
        fig = Figure(figsize=(6,4))
        canvas = FigureCanvas(fig)
        axes = [fig.add_subplot(len(self.gens),1,i+1)
                for i in range(len(self.gens))]
        for ax, (_, gen) in zip(axes, self.gens):
            ts,_ = gen.generate(self.state)
            data = getattr(ts,'data',np.array(ts))
            freqs, psd = signal.welch(
                data, self.sr, nperseg=256, noverlap=128
            )
            ax.semilogy(freqs, psd)
        axes[-1].set_xlabel("Frequency [Hz]")
        fig.tight_layout()
        return canvas

if __name__ == "__main__":
    cfg = {
        'sample_rate':      8000,
        'duration_seconds': 0.1,
        # …añade tus params de generators aquí…
    }
    app = QtWidgets.QApplication(sys.argv)
    win = GeneratorTabs(cfg, state=3)
    sys.exit(app.exec_())
