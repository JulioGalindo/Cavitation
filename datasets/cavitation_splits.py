from typing import Dict
import torch
from torch.utils.data import random_split
from .cavitation_dataset import CavitationDataset

def load_cavitation_datasets(
    wav_dir: str,
    metadata_csv: str,
    train_fraction: float = 0.8,
    seed: int = 42,
    transform=None
) -> Dict[str, CavitationDataset]:
    """
    Load the full CavitationDataset and split into train/val subsets.

    Returns
    -------
    dict
        {'train': Dataset, 'val': Dataset}
    """
    full = CavitationDataset(wav_dir, metadata_csv, transform=transform)
    n_total = len(full)
    n_train = int(n_total * train_fraction)
    n_val   = n_total - n_train
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full, [n_train, n_val], generator=generator)
    return {'train': train_ds, 'val': val_ds}
