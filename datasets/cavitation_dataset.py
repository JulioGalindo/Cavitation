import os
import csv
import soundfile as sf
import torch
from torch.utils.data import Dataset
from utils.timeseries import TimeSeries

class CavitationDataset(Dataset):
    """
    A PyTorch Dataset for cavitation audio segments and their labels.  
    Each item is a tuple (signal: torch.Tensor, cavitation_level: int).

    Parameters
    ----------
    root_dir : str
        Base directory for WAV files; paths in CSV are relative to this.
    metadata_csv : str
        Path to CSV with columns: wav_file, time_start, time_end, cavitation_level.
    transform : callable, optional
        Function applied to the torch.Tensor signal after slicing.
    """
    def __init__(self, root_dir: str, metadata_csv: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.entries = []
        with open(metadata_csv, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                path = os.path.join(root_dir, row['wav_file'])
                self.entries.append({
                    'path': path,
                    'start': float(row['time_start']),
                    'end':   float(row['time_end']),
                    'label': int(row['cavitation_level'])
                })

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        # Load entire file once
        data, sr = sf.read(entry['path'], dtype='float32')
        ts = TimeSeries(data, sr)
        # Slice in time
        ts = ts.slice_time(entry['start'], entry['end'])
        # Convert to tensor and add channel dim
        sig = torch.from_numpy(ts.data).float().unsqueeze(0)  # shape: [1, length]
        if self.transform:
            sig = self.transform(sig)
        return sig, entry['label']


def build_synthetic_dataset(config_path: str) -> CavitationDataset:
    """
    Generate synthetic cavitation WAVs & metadata, then return a dataset over them.
    """
    import yaml
    from data_generator import SyntheticDataGenerator

    cfg = yaml.safe_load(open(config_path))['data_generator']
    gen = SyntheticDataGenerator(config_path)
    out_dir  = cfg['default_output_dir']
    csv_path = cfg['default_csv_path']
    max_state = cfg.get('max_state', 6)
    # Generate WAVs for all states
    for state in range(max_state + 1):
        gen.generate_and_save(state)
    # Dataset expects WAVs in out_dir/wavs
    wav_dir = os.path.join(out_dir, 'wavs')
    return CavitationDataset(wav_dir, csv_path)
