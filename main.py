import torch
import yaml
from datasets import load_cavitation_datasets
from model import get_model
from trainer import Trainer


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def select_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def main(config_path='config.yaml'):
    config = load_config(config_path)
    device = select_device()

    print(f"[INFO] Using device: {device}")

    train_ds, val_ds = load_cavitation_datasets(config)
    model = get_model(config)

    trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        config=config,
        device=device
    )

    trainer.train()
    trainer.visualizer.finalize()


if __name__ == '__main__':
    main()
