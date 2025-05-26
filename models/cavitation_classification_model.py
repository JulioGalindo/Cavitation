# models/cavitation_classification_model.py

import torch
import torch.nn as nn
from utils.device_utils import configure_global, get_device

# ensure global optimizations (MPS precision, CPU threading)
configure_global()


class CavitationClassificationModel(nn.Module):
    """
    1D CNN for multiclass cavitation severity classification.
    """
    def __init__(
        self,
        input_channels: int = 1,
        num_filters: int = 32,
        kernel_size: int = 3,
        num_classes: int = 7,
    ):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(input_channels, num_filters, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(num_filters, num_filters * 2, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        # flatten and project to num_classes
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear((num_filters * 2) * 1, 128),  # adjust “1” if temporal dimension known
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, time) or (batch, 1, time)
        Returns:
            logits of shape (batch, num_classes)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = x.to(next(self.parameters()).device)
        feats = self.features(x)
        return self.classifier(feats)


def build_classification_model(cfg: dict) -> nn.Module:
    """
    Instantiate CavitationClassificationModel on the best device,
    optionally Torch-Compile when not on MPS.
    """
    device = get_device()
    model = CavitationClassificationModel(
        input_channels=1,
        num_filters=cfg.get('cls_filters', 32),
        kernel_size=cfg.get('cls_kernel_size', 3),
        num_classes=cfg.get('max_state', 6) + 1,
    ).to(device)

    # compile for speed on CUDA/CPU
    if hasattr(torch, 'compile') and device.type != 'mps':
        model = torch.compile(model)

    return model
