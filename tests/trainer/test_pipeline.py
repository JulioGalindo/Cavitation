import os
import torch
import yaml
import numpy as np
import soundfile as sf
import pandas as pd
from datasets import load_cavitation_datasets
from model import get_model
from trainer import Trainer


def generate_dummy_data(output_dir, duration_sec=2.0, sr=44100):
    os.makedirs(output_dir, exist_ok=True)
    wav_path = os.path.join(output_dir, "dummy.wav")
    csv_path = os.path.join(output_dir, "metadata.csv")

    signal = 0.05 * np.random.randn(int(sr * duration_sec))
    sf.write(wav_path, signal, sr, format='WAV', subtype='FLOAT')

    metadata = pd.DataFrame([{
        "wav_file": wav_path,
        "time_start": 0.0,
        "time_end": duration_sec,
        "cavitation_level": 3
    }])
    metadata.to_csv(csv_path, index=False)
    return csv_path


def test_pipeline():
    # load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # generate dummy dataset
    test_dir = "output/test_dummy"
    csv_path = generate_dummy_data(
        test_dir,
        duration_sec=config["data_generator"]["duration_seconds"]
    )

    # extract datasets from CSV and directory
    wav_dir = test_dir
    train_ds, val_ds = load_cavitation_datasets(
        wav_dir,
        metadata_csv=csv_path
    )

    # model and trainer
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else
        "cpu"
    )
    model = get_model(config).to(device)
    trainer = Trainer(model, train_ds, val_ds, config, device)

    # run training
    trainer.train()
    trainer.visualizer.finalize()
    print("✅ Pipeline ran successfully with dummy data.")


if __name__ == "__main__":
    test_pipeline()
