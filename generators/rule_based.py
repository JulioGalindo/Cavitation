import numpy as np
from generators.base import GenerativeModel
from utils.timeseries import TimeSeries
from layers.noise import NoiseLayer
from layers.burst import BurstLayer
from layers.tone import ToneLayer

class RuleBasedGenerator(GenerativeModel):
    """
    Rule-based synthetic cavitation signal generator.

    Combines noise, burst, and tonal components according to configuration.
    """
    def __init__(self, config: dict):
        self.sample_rate = config['sample_rate']
        self.duration_seconds = config['duration_seconds']
        self.num_samples = int(self.sample_rate * self.duration_seconds)

        noise_cfg = {
            'snr_db': config.get('snr_db', 30.0),
            'add_pink': config.get('add_pink', False)
        }
        burst_cfg = {
            'burst_count_mul': config.get('burst_count_mul', 6.0),
            'burst_width_range': config.get('burst_width_range', [0.0005, 0.002]),
            'burst_amp_range': config.get('burst_amp_range', [0.4, 1.0]),
            'envelope_type': config.get('envelope_type', 'gaussian'),
            'max_state': config.get('max_state', 6)
        }
        tone_cfg = {
            'add_tonal': config.get('add_tonal', True),
            'rotor_freq': config.get('rotor_freq', 50.0),
            'number_harmonics': config.get('number_harmonics', 1),
            'tone_amplitudes': config.get('tone_amplitudes', None),
            'mod_index': config.get('mod_index', 0.0)
        }
        self.noise_layer = NoiseLayer(sample_rate=self.sample_rate, num_samples=self.num_samples, **noise_cfg)
        self.burst_layer = BurstLayer(sample_rate=self.sample_rate, num_samples=self.num_samples, **burst_cfg)
        self.tone_layer = ToneLayer(sample_rate=self.sample_rate, num_samples=self.num_samples, **tone_cfg)

    def train(self, *args, **kwargs):
        return None

    def generate(self, state: int):
        """
        Generate a synthetic TimeSeries and metadata for a given cavitation state.

        Returns
        -------
        ts : TimeSeries
            Generated signal with time vector.
        metadata : list of dict
            Each dict contains 'time_start', 'time_end', 'state'.
        """
        noise = self.noise_layer.generate()
        burst_signal, metadata = self.burst_layer.generate(state)
        tone_signal = self.tone_layer.generate()
        signal = noise + burst_signal + tone_signal
        ts = TimeSeries(data=signal, sample_rate=self.sample_rate)
        return ts, metadata
