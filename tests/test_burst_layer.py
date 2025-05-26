import numpy as np
import pytest
from layers.burst import BurstLayer

@pytest.mark.parametrize("state", [0, 1, 3])
def test_burst_output_and_metadata(state):
    layer = BurstLayer(sample_rate=1000, num_samples=1000,
                       burst_count_mul=2.0, burst_width_range=[0.01, 0.02],
                       burst_amp_range=[0.5, 1.0], envelope_type='gaussian', max_state=6)
    bursts, metadata = layer.generate(state=state)
    assert isinstance(bursts, np.ndarray)
    assert bursts.shape == (1000,)
    if state == 0:
        assert np.allclose(bursts, 0)
        assert metadata == []
    else:
        expected_count = int(round(2.0 * state))
        assert len(metadata) == expected_count
        for entry in metadata:
            assert 0 <= entry['time_start'] < entry['time_end'] <= 1.0
            assert entry['state'] == state
