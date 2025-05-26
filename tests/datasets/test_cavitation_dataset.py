import os
import csv
import numpy as np
import torch
import soundfile as sf
import pytest

from datasets.cavitation_dataset import CavitationDataset

@pytest.fixture
def tmp_cavitation_dir(tmp_path):
    """
    Create a temporary directory containing:
     - a single WAV file of a known half‐second sine burst
     - a metadata CSV pointing to it with time_start=0, time_end=0.5, cavitation_level=3
    """
    # parameters
    sr = 8000
    duration = 0.5
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    data = 0.5 * np.sin(2 * np.pi * 5 * t).astype(np.float32)
    # make dirs
    wavs = tmp_path / "wavs"
    wavs.mkdir()
    wav_file = wavs / "burst.wav"
    # write float32 WAV
    sf.write(str(wav_file), data, sr, format="WAV", subtype="FLOAT")
    # metadata
    csv_file = tmp_path / "meta.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["wav_file", "time_start", "time_end", "cavitation_level"])
        writer.writerow(["burst.wav", "0.0", str(duration/2), "3"])
    return str(wavs), str(csv_file), data, sr, duration

def test_len_and_indexing(tmp_cavitation_dir):
    wav_dir, csv_path, data, sr, duration = tmp_cavitation_dir
    ds = CavitationDataset(wav_dir, csv_path)
    # only one entry
    assert len(ds) == 1

    sig_tensor, label = ds[0]
    # correct types and label
    assert isinstance(sig_tensor, torch.Tensor)
    assert label == 3

    # half‐second slice → quarter‐second so samples = sr * duration/2
    expected = int(sr * (duration/2))
    assert sig_tensor.shape[0] == expected

    # compare first few samples exactly
    np.testing.assert_allclose(sig_tensor[:5].numpy(), data[:5], rtol=0, atol=0)

def test_transform_applied(tmp_cavitation_dir):
    wav_dir, csv_path, data, sr, duration = tmp_cavitation_dir

    def double(x: torch.Tensor):
        return x * 2

    ds = CavitationDataset(wav_dir, csv_path, transform=double)
    sig_tensor, _ = ds[0]
    expected = data[:int(sr * (duration/2))] * 2
    np.testing.assert_allclose(sig_tensor.numpy(), expected, rtol=0, atol=0)
