import os
import yaml
import pytest
import soundfile as sf
from data_generator import SyntheticDataGenerator

@pytest.mark.parametrize("state", [0, 2])
def test_generate_and_save(tmp_path, state):
    # Prepare temporary config YAML
    out_dir = tmp_path / "out"
    csv_path = out_dir / "metadata.csv"
    cfg = {
        'data_generator': {
            'method': 'rule_based',
            'sample_rate': 100,
            'duration_seconds': 0.1,
            'snr_db': 10,
            'add_pink': False,
            'burst_count_mul': 1.0,
            'burst_width_range': [0.001, 0.002],
            'burst_amp_range': [0.5, 1.0],
            'envelope_type': 'gaussian',
            'add_tonal': False,
            'default_output_dir': str(out_dir),
            'default_csv_path': str(csv_path)
        }
    }
    config_file = tmp_path / "config.yaml"
    with open(config_file, 'w') as f:
        yaml.safe_dump(cfg, f)

    # Generate data
    gen = SyntheticDataGenerator(str(config_file))
    wav_path, csv_file = gen.generate_and_save(state)

    # Check WAV file exists and properties
    assert os.path.exists(wav_path)
    data, sr = sf.read(wav_path)
    expected_samples = int(cfg['data_generator']['sample_rate'] * cfg['data_generator']['duration_seconds'])
    assert sr == cfg['data_generator']['sample_rate']
    assert data.shape[0] == expected_samples

    # Check CSV file exists and content
    assert os.path.exists(csv_file)
    with open(csv_file, 'r') as f:
        lines = f.readlines()
    # Header + at most one data line when state=0, more otherwise
    assert 'wav_file' in lines[0]
    if state > 0:
        assert len(lines) > 1
    else:
        assert len(lines) == 1
