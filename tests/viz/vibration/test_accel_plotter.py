# tests/viz/vibration/test_accel_plotter.py
import numpy as np
import matplotlib.pyplot as plt
import pytest
from visualization.vibration.accel_plotter import AccelPlotter

@pytest.fixture
def sine_data():
    sr = 1000
    t = np.linspace(0, 1, sr, endpoint=False)
    data = np.sin(2 * np.pi * 5 * t)  # 5 Hz sine
    return data, sr

def test_accel_plotter(sine_data):
    data, sr = sine_data
    plotter = AccelPlotter(sample_rate=sr)
    fig, ax = plt.subplots()
    ax_out = plotter.plot(data, ax=ax, color='g', title='Accel Test', ylabel='g')
    assert ax_out is ax
    assert ax.get_title() == 'Accel Test'
    assert ax.get_xlabel() == 'Time [s]'
    assert ax.get_ylabel() == 'g'

    lines = ax.get_lines()
    assert len(lines) == 1
    x_data, y_data = lines[0].get_data()
    # check a few sample points
    for i in [0, sr//4, sr//2, 3*sr//4]:
        assert y_data[i] == pytest.approx(data[i], rel=1e-6)
        assert x_data[i] == pytest.approx(i/sr, rel=1e-6)
    plt.close(fig)
