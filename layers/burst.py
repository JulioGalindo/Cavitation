import numpy as np
from layers.envelope import get_envelope

class BurstLayer:
    """
    Generates burst events (simulating cavitation) with configurable envelopes.
    """
    def __init__(self, sample_rate: int, num_samples: int,
                 burst_count_mul: float = 6.0,
                 burst_width_range: list = [0.0005, 0.002],
                 burst_amp_range: list = [0.4, 1.0],
                 envelope_type: str = 'gaussian',
                 max_state: int = 6):
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        self.sample_rate = sample_rate
        self.num_samples = num_samples
        self.burst_count_mul = burst_count_mul
        self.burst_width_range = burst_width_range
        self.burst_amp_range = burst_amp_range
        self.envelope_type = envelope_type
        self.max_state = max_state

    def train(self, *args, **kwargs):
        # No training required for rule-based bursts
        return None

    def generate(self, state: int):
        """
        Generate burst component for given cavitation state.

        Parameters
        ----------
        state : int
            Cavitation severity level (0=no cavitation to max_state).

        Returns
        -------
        bursts : np.ndarray
            Burst signal array of length num_samples.
        metadata : list of dict
            Each dict contains 'time_start', 'time_end', 'state'.
        """
        bursts = np.zeros(self.num_samples, dtype=np.float32)
        metadata = []
        # Determine number of bursts proportional to state
        num_bursts = int(round(self.burst_count_mul * state))
        for _ in range(num_bursts):
            # Random burst duration
            width_sec = float(np.random.uniform(
                self.burst_width_range[0], self.burst_width_range[1]))
            width_samp = max(1, int(round(width_sec * self.sample_rate)))
            # Random start index ensuring burst fits
            max_start = max(0, self.num_samples - width_samp)
            start_idx = int(np.random.randint(0, max_start + 1))
            # Create envelope via factory
            envelope = get_envelope(self.envelope_type, width_samp).astype(np.float32)
            # Random amplitude scaled by state
            base_amp = float(np.random.uniform(
                self.burst_amp_range[0], self.burst_amp_range[1]))
            amp = base_amp * (state / self.max_state)
            # Add burst to signal
            bursts[start_idx:start_idx + width_samp] += amp * envelope
            # Record metadata (in seconds)
            metadata.append({
                'time_start': start_idx / self.sample_rate,
                'time_end': (start_idx + width_samp) / self.sample_rate,
                'state': state
            })
        return bursts, metadata
