import numpy as np
from abc import ABC, abstractmethod

class EnvelopeGenerator(ABC):
    """
    Abstract base for envelope generators used in BurstLayer.
    """
    @abstractmethod
    def generate(self, length: int) -> np.ndarray:
        """
        Generate an envelope array of specified length.

        Parameters
        ----------
        length : int
            Number of samples for the envelope.

        Returns
        -------
        np.ndarray
            Envelope array normalized to peak=1.
        """
        pass

class GaussianEnvelope(EnvelopeGenerator):
    def generate(self, length: int) -> np.ndarray:
        t = np.arange(length)
        center = (length - 1) / 2.0
        sigma = length / 6.0
        env = np.exp(-0.5 * ((t - center) / sigma) ** 2)
        return env / np.max(env) if env.max() > 0 else env

class ExponentialEnvelope(EnvelopeGenerator):
    def generate(self, length: int) -> np.ndarray:
        t = np.arange(length)
        center = (length - 1) / 2.0
        scale = length / 5.0
        env = np.exp(-np.abs(t - center) / scale)
        return env / np.max(env) if env.max() > 0 else env

class SincEnvelope(EnvelopeGenerator):
    def generate(self, length: int) -> np.ndarray:
        t = np.arange(length)
        center = (length - 1) / 2.0
        arg = (t - center) / (length / 10.0)
        env = np.sinc(arg)
        return env / np.max(np.abs(env)) if np.max(np.abs(env)) > 0 else env

class RectangularEnvelope(EnvelopeGenerator):
    def generate(self, length: int) -> np.ndarray:
        return np.ones(length, dtype=np.float32)

class TriangularEnvelope(EnvelopeGenerator):
    def generate(self, length: int) -> np.ndarray:
        t = np.arange(length)
        center = (length - 1) / 2.0
        env = 1.0 - np.abs((t - center) / center)
        return env / np.max(env) if env.max() > 0 else env

class CosineEnvelope(EnvelopeGenerator):
    def generate(self, length: int) -> np.ndarray:
        t = np.arange(length)
        env = 0.5 * (1.0 + np.cos(2 * np.pi * (t - (length - 1) / 2.0) / length))
        return env / np.max(env) if env.max() > 0 else env


def get_envelope(env_type: str, length: int) -> np.ndarray:
    """
    Factory function to obtain envelope array by type.

    Parameters
    ----------
    env_type : str
        One of 'gaussian', 'exponential', 'sinc', 'rectangular', 'triangular', 'cosine'.
    length : int
        Number of samples for the envelope.

    Returns
    -------
    np.ndarray
        Envelope array normalized to peak=1.
    """
    env_type = env_type.lower()
    if env_type == 'gaussian':
        return GaussianEnvelope().generate(length)
    elif env_type == 'exponential':
        return ExponentialEnvelope().generate(length)
    elif env_type == 'sinc':
        return SincEnvelope().generate(length)
    elif env_type == 'rectangular':
        return RectangularEnvelope().generate(length)
    elif env_type == 'triangular':
        return TriangularEnvelope().generate(length)
    elif env_type == 'cosine':
        return CosineEnvelope().generate(length)
    else:
        raise ValueError(f"Unsupported envelope type: {env_type}")
