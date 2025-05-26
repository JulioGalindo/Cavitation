import matplotlib.pyplot as plt
from visualization.base import BasePlotter

class GridManager:
    """
    Arrange multiple plots in a grid layout.
    """
    def __init__(self, wav_path=None, vibration_data=None):
        self.wav_path = wav_path
        self.vibration_data = vibration_data

    def plot_grid(self, configs, nrows=1, ncols=1, fig_title=None, figsize=(10, 6)):
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

        for ax, cfg in zip(axes, configs):
            plotter = cfg['plotter']
            if not isinstance(plotter, BasePlotter):
                raise ValueError('plotter must extend BasePlotter')
            source = cfg.get('source_data')
            if source is None and self.wav_path:
                ax_out = plotter.plot(self.wav_path, ax=ax, **cfg.get('plot_kwargs', {}))
            else:
                ax_out = plotter.plot(source, ax=ax, **cfg.get('plot_kwargs', {}))
            ax_out.set_title(cfg.get('title', ax_out.get_title()))
            desc = cfg.get('description')
            if desc:
                ax_out.text(0.5, -0.1, desc, transform=ax_out.transAxes,
                            ha='center', va='top')

        if fig_title:
            fig.suptitle(fig_title)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        return fig, axes
