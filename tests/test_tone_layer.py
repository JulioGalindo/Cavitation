import numpy as np
import pytest
from layers.tone import ToneLayer

def test_no_tone_when_disabled():
    layer = ToneLayer(sample_rate=1000, num_samples=1000, add_tonal=False)
    tone = layer.generate()
    assert isinstance(tone, np.ndarray)
    assert tone.shape == (1000,)
    assert np.allclose(tone, 0)

@pytest.mark.parametrize("harmonics, amplitudes, mod", [
    (1, [1.0], 0.0),
    (2, [0.5, 0.25], 0.1),
    (3, [0.2, 0.2, 0.2], 0.5)
])
def test_tone_properties(harmonics, amplitudes, mod):
    layer = ToneLayer(sample_rate=1000, num_samples=1000,
                      add_tonal=True, rotor_freq=5.0,
                      number_harmonics=harmonics,
                      tone_amplitudes=amplitudes,
                      mod_index=mod)
    tone = layer.generate()
    assert tone.shape == (1000,)
    # Check normalization
    assert np.isclose(np.max(np.abs(tone)), 1.0, atol=1e-6)
