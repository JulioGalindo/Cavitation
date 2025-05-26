import numpy as np
from generators.rule_based import RuleBasedGenerator
from utils.timeseries import TimeSeries


def get_cfg():
    return {
        'sample_rate': 200,
        'duration_seconds': 0.5,
        'snr_db': 20,
        'add_pink': False,
        'burst_count_mul': 2.0,
        'burst_width_range': [0.01, 0.02],
        'burst_amp_range': [0.5, 1.0],
        'envelope_type': 'gaussian',
        'add_tonal': True,
        'rotor_freq': 5.0,
        'number_harmonics': 2,
        'tone_amplitudes': [0.3, 0.1],
        'mod_index': 0.1
    }


def test_rule_based_generate_signal_and_metadata():
    cfg = get_cfg()
    gen = RuleBasedGenerator(cfg)
    ts, metadata = gen.generate(state=1)
    assert isinstance(ts, TimeSeries)
    # Length in samples
    assert len(ts) == int(cfg['sample_rate'] * cfg['duration_seconds'])
    # Data not all zeros for state=1 with tonal
    assert not np.allclose(ts.data, 0)
    # Metadata validation
    assert isinstance(metadata, list)
    for entry in metadata:
        assert 'time_start' in entry and 'time_end' in entry and 'state' in entry
        assert 0 <= entry['time_start'] < entry['time_end'] <= cfg['duration_seconds']
        assert entry['state'] == 1
