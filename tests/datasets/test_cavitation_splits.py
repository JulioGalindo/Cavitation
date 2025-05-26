import torch
import numpy as np
import csv
import os
import pytest
from torch.utils.data import Subset

from datasets.cavitation_splits import load_cavitation_datasets
from datasets.cavitation_dataset import CavitationDataset

@pytest.fixture
def full_dataset(tmp_path):
    # Create two WAVs and a metadata file with two entries
    sr = 1000
    dur = 0.2
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    data1 = np.zeros_like(t, dtype=np.float32)
    data2 = np.ones_like(t, dtype=np.float32)

    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir()
    sf1 = wav_dir / "a.wav"
    sf2 = wav_dir / "b.wav"
    # use float16 subtype just to vary; loader should still read float32
    import soundfile as sf
    sf.write(str(sf1), data1, sr, format="WAV", subtype="FLOAT")
    sf.write(str(sf2), data2, sr, format="WAV", subtype="FLOAT")

    meta = tmp_path / "meta.csv"
    with open(meta, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["wav_file", "time_start", "time_end", "cavitation_level"])
        writer.writerow(["a.wav", "0.0", str(dur), "0"])
        writer.writerow(["b.wav", "0.0", str(dur), "1"])

    return str(wav_dir), str(meta)

def test_splits_and_reproducibility(full_dataset):
    wav_dir, csv_path = full_dataset
    splits1 = load_cavitation_datasets(wav_dir, csv_path, train_fraction=0.5, seed=123)
    splits2 = load_cavitation_datasets(wav_dir, csv_path, train_fraction=0.5, seed=123)
    # both should be Subset-backed datasets
    assert isinstance(splits1["train"], Subset)
    assert isinstance(splits1["val"], Subset)
    # lengths should sum to full
    total_len = len(splits1["train"]) + len(splits1["val"])
    assert total_len == len(CavitationDataset(wav_dir, csv_path))

    # same seed → same indices
    assert splits1["train"].indices == splits2["train"].indices
    assert splits1["val"].indices == splits2["val"].indices

def test_split_ratios(full_dataset):
    wav_dir, csv_path = full_dataset
    splits = load_cavitation_datasets(wav_dir, csv_path, train_fraction=0.25, seed=0)
    # floor(2 * .25) = 0 train, 2 val
    assert len(splits["train"]) == 0
    assert len(splits["val"]) == 2
