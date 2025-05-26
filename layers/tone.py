import numpy as np

class ToneLayer:
    """
    Generates tonal (harmonic) signal components based on rotor frequency.
    """
    def __init__(self, sample_rate: int, num_samples: int,
                 add_tonal: bool = True,
                 rotor_freq: float = 50.0,
                 number_harmonics: int = 1,
                 tone_amplitudes: list = None,
                 mod_index: float = 0.0):
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        self.sample_rate = sample_rate
        self.num_samples = num_samples
        self.add_tonal = add_tonal
        self.rotor_freq = rotor_freq
        self.number_harmonics = number_harmonics
        self.mod_index = mod_index
        # Default amplitude list: uniform small amplitude
        if tone_amplitudes is None:
            self.tone_amplitudes = [1.0 / number_harmonics] * number_harmonics
        else:
            if len(tone_amplitudes) < number_harmonics:
                # pad with last amplitude
                self.tone_amplitudes = tone_amplitudes + \
                    [tone_amplitudes[-1]] * (number_harmonics - len(tone_amplitudes))
            else:
                self.tone_amplitudes = tone_amplitudes

    def generate(self) -> np.ndarray:
        """
        Generate the tonal component.

        Returns
        -------
        np.ndarray
            Tone signal array of length num_samples.
        """
        if not self.add_tonal or self.number_harmonics <= 0:
            return np.zeros(self.num_samples, dtype=np.float32)

        # Time vector
        t = np.arange(self.num_samples) / self.sample_rate
        tone = np.zeros_like(t, dtype=np.float32)
        # Phase modulation term
        if self.mod_index and self.rotor_freq > 0:
            modulation = self.mod_index * np.sin(2 * np.pi * self.rotor_freq * t)
        else:
            modulation = 0.0

        # Sum harmonics
        for k in range(1, self.number_harmonics + 1):
            amp = float(self.tone_amplitudes[k - 1])
            phase = 2 * np.pi * k * self.rotor_freq * t
            # Apply modulation to phase
            if isinstance(modulation, np.ndarray):
                phase = phase + modulation
            elif modulation != 0.0:
                phase = phase + modulation
            tone += amp * np.sin(phase)

        # Normalize to avoid excessive amplitude
        max_val = np.max(np.abs(tone))
        if max_val > 0:
            tone = tone / max_val
        return tone.astype(np.float32)
