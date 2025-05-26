# tests/models/test_cavitation_classification_model.py

import torch
import numpy as np
import pytest
from models.cavitation_classification_model import build_classification_model

@pytest.fixture
def cfg():
    return {
        'cls_filters': 8,
        'cls_kernel_size': 3,
        'sample_rate': 50,
        'duration_seconds': 0.1,
        'max_state': 4,   # so num_classes = 5
    }

def test_classification_model_shapes(cfg):
    model = build_classification_model(cfg)
    device = next(model.parameters()).device

    # batch of 3 signals
    length = int(cfg['sample_rate'] * cfg['duration_seconds'])
    x = torch.randn(3, length, device=device)
    logits = model(x)

    # output shape should be (batch, num_classes)
    assert logits.shape == (3, cfg['max_state'] + 1)
    # logits can be any real, but no NaNs
    assert torch.isfinite(logits).all()

def test_classification_model_cpu_and_compile(monkeypatch, cfg):
    import torch
    # Force CPU only
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    model = build_classification_model(cfg)
    assert next(model.parameters()).device.type == 'cpu'
    # should still respond to forward
    dummy = torch.zeros(1, int(cfg['sample_rate']*cfg['duration_seconds']))
    out = model(dummy)
    assert out.shape == (1, cfg['max_state'] + 1)
