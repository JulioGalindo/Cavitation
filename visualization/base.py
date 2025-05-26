from abc import ABC, abstractmethod

class BasePlotter(ABC):
    """
    Abstract base class for all plotters.

    Subclasses must implement `plot(wav_path, ax=None, **plot_kwargs)`.
    """
    @abstractmethod
    def plot(self, wav_path: str, ax=None, **plot_kwargs):
        """
        Plot the relevant analysis for a WAV file or other data source.

        Parameters
        ----------
        wav_path : str
            Path to the .wav file to analyze.
        ax : matplotlib.axes.Axes, optional
            Axes on which to draw. If None, a new figure and axes are created.
        plot_kwargs : dict
            Additional keyword arguments for plotting (e.g., color, title).

        Returns
        -------
        matplotlib.axes.Axes
            The axes with the plot.
        """
        pass
