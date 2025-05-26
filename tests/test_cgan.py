import torch
import numpy as np
import pytest
from generators.cgan import CGANGenerator
from utils.timeseries import TimeSeries

# Dummy dataset for testing
class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples, signal_length, max_state):
        self.num_samples = num_samples
        self.signal_length = signal_length
        self.max_state = max_state

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # random signal and corresponding label
        x = torch.randn(self.signal_length)
        label = idx % (self.max_state + 1)
        return x, label

# Configuration for CGANGenerator
def get_cfg():
    return {
        'sample_rate': 100,
        'duration_seconds': 0.1,
        'latent_dim': 10,
        'hidden_dim': 16,
        'epochs': 1,
        'batch_size': 2,
        'lr': 1e-3,
        'max_state': 3
    }

# Test training loop runs without errors and generate returns valid output
def test_cgan_train_and_generate():
    cfg = get_cfg()
    gen = CGANGenerator(cfg)
    signal_length = int(cfg['sample_rate'] * cfg['duration_seconds'])
    dataset = DummyDataset(num_samples=4, signal_length=signal_length, max_state=cfg['max_state'])

    # Training should complete without exceptions
    gen.train(dataset)

    # Generation for each state
    for state in range(cfg['max_state'] + 1):
        ts, metadata = gen.generate(state)
        # Check return types
        assert isinstance(ts, TimeSeries)
        assert isinstance(metadata, list)
        # Check signal length and values
        assert len(ts) == signal_length
        assert float(np.max(np.abs(ts.data))) == pytest.approx(1.0, rel=1e-6)
        # Metadata should be empty list
        assert metadata == []
