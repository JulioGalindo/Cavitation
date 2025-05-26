import numpy as np
import pytest
from layers.noise import NoiseLayer


def test_white_noise_properties():
    layer = NoiseLayer(sample_rate=1000, num_samples=1000, snr_db=10, add_pink=False)
    noise = layer.generate()
    assert isinstance(noise, np.ndarray)
    assert noise.shape == (1000,)
    assert noise.dtype == np.float32

@pytest.mark.parametrize("snr_db", [0, 10, 20, 40])
def test_calibrate_to_snr(snr_db):
    layer = NoiseLayer(sample_rate=1000, num_samples=1000, snr_db=snr_db, add_pink=False)
    signal = np.ones(1000, dtype=np.float32)
    noise = layer.generate()
    scaled = layer.calibrate_to_snr(signal, noise)
    ps = np.mean(signal**2)
    pn = np.mean(scaled**2)
    achieved_snr = 10 * np.log10(ps / pn)
    assert pytest.approx(achieved_snr, rel=1e-1) == snr_db
