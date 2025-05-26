from abc import ABC, abstractmethod

class GenerativeModel(ABC):
    """
    Abstract base class for all synthetic data generative models.

    Defines the interface that all concrete generators must implement.
    """

    @abstractmethod
    def train(self, *args, **kwargs):
        """
        Train the model (only for ML-based generators).

        Parameters
        ----------
        *args, **kwargs
            Model-specific training arguments.
        """
        pass

    @abstractmethod
    def generate(self, state: int) -> "TimeSeries":
        """
        Generate a synthetic cavitation signal conditioned on the given state.

        Parameters
        ----------
        state : int
            Cavitation severity level (e.g., 0=no cavitation, up to 6=max severity).

        Returns
        -------
        TimeSeries
            Generated signal with associated time vector.
        """
        pass
