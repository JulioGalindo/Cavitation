#!/usr/bin/env python3
import sys
import numpy as np
import scipy.signal as signal
import librosa
import librosa.display

from PyQt5 import QtWidgets
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar
)
from matplotlib.figure import Figure

from generators.rule_based import RuleBasedGenerator
from generators.physical import PhysicalModelGenerator
from generators.cgan import CGANGenerator
from generators.cvae import CVAEGenerator
from generators.diffusion import DiffusionGenerator

class GeneratorStack(QtWidgets.QMainWindow):
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

        # Buttons to switch pages
        btn_layout = QtWidgets.QHBoxLayout()
        for idx, name in enumerate(["Waveforms", "Mel Spec", "STFT Spec", "PSD"]):
            btn = QtWidgets.QPushButton(name)
            btn.clicked.connect(lambda _, i=idx: self.stack.setCurrentIndex(i))
            btn_layout.addWidget(btn)

        # StackedLayout
        self.stack = QtWidgets.QStackedLayout()
        for builder in (self._make_waveform_page,
                        self._make_mel_page,
                        self._make_stft_page,
                        self._make_psd_page):
            self.stack.addWidget(builder())

        # Main layout
        container = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout(container)
        main_layout.addLayout(btn_layout)
        main_layout.addLayout(self.stack)
        self.setCentralWidget(container)
        self.resize(900, 700)
        self.show()

    def _make_canvas_with_toolbar(self, nrows):
        """Helper: returns (page_widget, axes_list)."""
        fig = Figure(figsize=(6,4))
        canvas = FigureCanvas(fig)
        toolbar = NavigationToolbar(canvas, self)
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.addWidget(toolbar)
        layout.addWidget(canvas)
        axes = [fig.add_subplot(nrows, 1, i+1) for i in range(nrows)]
        return page, axes

    def _make_waveform_page(self):
        page, axes = self._make_canvas_with_toolbar(len(self.gens))
        for ax, (name, gen) in zip(axes, self.gens):
            ts,_ = gen.generate(self.state)
            data = getattr(ts, 'data', np.array(ts))
            ax.plot(self.t, data, lw=1)
            ax.set_ylabel(name)
        axes[-1].set_xlabel("Time [s]")
        axes[0].set_title("Waveforms")
        page.layout().itemAt(0).widget().setToolTip("Waveform toolbar")
        return page

    def _make_mel_page(self):
        page, axes = self._make_canvas_with_toolbar(len(self.gens))
        for ax, (_, gen) in zip(axes, self.gens):
            ts,_ = gen.generate(self.state)
            data = getattr(ts, 'data', np.array(ts))
            S = librosa.feature.melspectrogram(
                y=data, sr=self.sr, n_fft=256, hop_length=128, n_mels=64)
            S_db = librosa.power_to_db(S, ref=np.max)
            librosa.display.specshow(
                S_db, sr=self.sr, hop_length=128,
                x_axis=None, y_axis='mel',
                ax=ax, cmap='magma')
        axes[-1].set_xlabel("Time [s]")
        axes[0].set_title("Mel Spectrograms")
        return page

    def _make_stft_page(self):
        page, axes = self._make_canvas_with_toolbar(len(self.gens))
        for ax, (_, gen) in zip(axes, self.gens):
            ts,_ = gen.generate(self.state)
            data = getattr(ts, 'data', np.array(ts))
            f, tt, Sxx = signal.spectrogram(
                data, fs=self.sr, nperseg=128, noverlap=64)
            ax.pcolormesh(tt, f, 10*np.log10(Sxx+1e-10), shading='gouraud')
        axes[-1].set_xlabel("Time [s]")
        axes[0].set_title("STFT Spectrograms")
        return page

    def _make_psd_page(self):
        page, axes = self._make_canvas_with_toolbar(len(self.gens))
        for ax, (_, gen) in zip(axes, self.gens):
            ts,_ = gen.generate(self.state)
            data = getattr(ts, 'data', np.array(ts))
            freqs, psd = signal.welch(
                data, self.sr, nperseg=256, noverlap=128)
            ax.semilogy(freqs, psd)
        axes[-1].set_xlabel("Frequency [Hz]")
        axes[0].set_title("Power Spectral Densities")
        return page

if __name__ == "__main__":
    cfg = {
        'sample_rate':      8000,
        'duration_seconds': 0.1,
        # …tus parámetros de generators…
    }
    app = QtWidgets.QApplication(sys.argv)
    win = GeneratorStack(cfg, state=3)
    sys.exit(app.exec_())
