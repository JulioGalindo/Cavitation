import torch
import numpy as np
import pytest
from generators.diffusion import DiffusionGenerator
from utils.timeseries import TimeSeries

# Dummy dataset: returns (signal, label), label is ignored
class DummyDiffusionDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples, signal_length):
        self.num_samples = num_samples
        self.signal_length = signal_length

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Return a random signal and dummy label
        signal = torch.randn(self.signal_length)
        label = 0
        return signal, label

# Configuration fixture for DiffusionGenerator
test_cfg = {
    'sample_rate': 100,
    'duration_seconds': 0.1,
    'batch_size': 2,
    'epochs': 1,
    'lr': 1e-3,
    'timesteps': 10,
    'beta_start': 1e-4,
    'beta_end': 1e-3,
    'unet_channels': 8,
    'time_emb_dim': 32
}

@pytest.fixture
def diffusion_gen():
    return DiffusionGenerator(test_cfg)

def test_diffusion_train_and_generate(diffusion_gen):
    # Prepare dataset
    signal_len = int(test_cfg['sample_rate'] * test_cfg['duration_seconds'])
    dataset = DummyDiffusionDataset(num_samples=4, signal_length=signal_len)

    # Train should complete without error
    diffusion_gen.train(dataset)

    # Generate for arbitrary state (state ignored)
    ts, metadata = diffusion_gen.generate(state=0)
    assert isinstance(ts, TimeSeries)
    assert isinstance(metadata, list)
    assert metadata == []

    # Check length and finite values
    assert len(ts) == signal_len
    assert np.all(np.isfinite(ts.data))
    # Check normalization
    max_val = float(np.max(np.abs(ts.data)))
    assert max_val == pytest.approx(1.0, rel=1e-6)

def test_diffusion_device_and_buffers(diffusion_gen):
    # Verify that buffers betas and alphas_cum exist and on correct device
    dev = diffusion_gen.device
    assert hasattr(diffusion_gen, 'betas')
    assert hasattr(diffusion_gen, 'alphas_cum')
    assert diffusion_gen.betas.device.type == dev.type
    assert diffusion_gen.alphas_cum.device.type == dev.type

    # Check beta and alpha shapes
    assert diffusion_gen.betas.shape[0] == test_cfg['timesteps']
    assert diffusion_gen.alphas_cum.shape[0] == test_cfg['timesteps']
