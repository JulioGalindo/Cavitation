import numpy as np
import pytest
from generators.physical import PhysicalModelGenerator
from utils.timeseries import TimeSeries


def get_cfg():
    return {
        'sample_rate': 100,
        'duration_seconds': 0.2,
        'physical_model': {
            'rho': 998.0,
            'sigma': 0.072,
            'mu': 0.001,
            'P_v': 2300.0,
            'R0': 1.0,
            'dR0': 0.0,
            'scale_factor': 1.0,
            'rotor_freq': 0.0,
            'max_state': 5
        }
    }

@pytest.mark.parametrize("state", [0, 3, 5])
def test_physical_generate_states(state):
    cfg = get_cfg()
    gen = PhysicalModelGenerator(cfg)
    signal_length = int(cfg['sample_rate'] * cfg['duration_seconds'])

    ts, metadata = gen.generate(state)
    # Check types
    assert isinstance(ts, TimeSeries)
    assert isinstance(metadata, list)
    # Check correct length
    assert len(ts) == signal_length
    # Time vector monotonic from 0 to last sample
    t = ts.time
    assert t[0] == pytest.approx(0.0)
    expected_last = (signal_length - 1) / cfg['sample_rate']
    assert t[-1] == pytest.approx(expected_last, rel=1e-6)
    # Metadata should always be empty
    assert metadata == []
    # Signal values finite
    assert np.all(np.isfinite(ts.data))

@pytest.mark.parametrize("filter_cfg", [None, {'filter_type': 'butter', 'filter_order': 3, 'filter_cutoff': [10, 40]}])
def test_physical_filter_option(filter_cfg):
    cfg = get_cfg()
    if filter_cfg:
        cfg.update(filter_cfg)
    gen = PhysicalModelGenerator(cfg)
    ts, _ = gen.generate(state=2)
    # Ensure data is still finite after optional filtering
    assert np.all(np.isfinite(ts.data))
