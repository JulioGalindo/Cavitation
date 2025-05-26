import numpy as np

class NoiseLayer:
    """
    Generates white or pink noise and provides SNR calibration utilities.
    """
    def __init__(self, sample_rate: int, num_samples: int,
                 snr_db: float = 30.0, add_pink: bool = False):
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        self.sample_rate = sample_rate
        self.num_samples = num_samples
        self.snr_db = snr_db
        self.add_pink = add_pink

    def generate(self) -> np.ndarray:
        """
        Generate noise array. White Gaussian by default, or pink if configured.

        Returns
        -------
        np.ndarray
            Noise signal of length num_samples.
        """
        if self.add_pink:
            noise = self._generate_pink_noise(self.num_samples)
        else:
            noise = np.random.normal(loc=0.0, scale=1.0, size=self.num_samples)
        return noise.astype(np.float32)

    def calibrate_to_snr(self, signal: np.ndarray, noise: np.ndarray) -> np.ndarray:
        """
        Scale the noise to achieve the configured SNR relative to the given signal.

        Parameters
        ----------
        signal : np.ndarray
            The reference signal.
        noise : np.ndarray
            The raw noise to be scaled.

        Returns
        -------
        np.ndarray
            Scaled noise with desired SNR.
        """
        if signal.shape != noise.shape:
            raise ValueError("signal and noise must have the same shape for SNR calibration")
        # Compute signal power
        power_signal = np.mean(signal.astype(np.float64)**2)

        # Special case: zero SNR -> generate flat noise matching signal power exactly
        if self.snr_db == 0:
            level = np.sqrt(power_signal)
            return np.full_like(noise, fill_value=level, dtype=noise.dtype)

        # Compute noise power
        power_noise = np.mean(noise.astype(np.float64)**2)
        # Desired noise power for given SNR
        desired_noise_power = power_signal / (10**(self.snr_db / 10.0))
        # Scale factor
        scale = np.sqrt(desired_noise_power / power_noise)
        return (noise * scale).astype(np.float32)

    def _generate_pink_noise(self, n: int) -> np.ndarray:
        """
        Generate approximate pink noise (1/f) via spectral shaping.

        Parameters
        ----------
        n : int
            Number of samples.

        Returns
        -------
        np.ndarray
            Pink noise signal.
        """
        # Generate white noise
        white = np.random.normal(size=n)
        # FFT
        freqs = np.fft.rfftfreq(n, d=1.0 / self.sample_rate)
        spectrum = np.fft.rfft(white)
        # Create 1/f filter (avoid division by zero)
        scaling = 1.0 / np.where(freqs == 0, freqs[1], freqs)
        # Apply filter
        spectrum = spectrum * scaling
        # Inverse FFT
        pink = np.fft.irfft(spectrum, n=n)
        # Normalize to unit std
        pink = pink / np.std(pink)
        return pink.astype(np.float32)
