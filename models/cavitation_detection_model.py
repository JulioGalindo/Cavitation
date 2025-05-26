# models/cavitation_detection_model.py

import torch
import torch.nn as nn
from utils.device_utils import configure_global, get_device

# Apply global optimizations (MPS precision, CPU threading) immediately
configure_global()


class CavitationDetectionModel(nn.Module):
    """
    1D CNN for binary cavitation detection.
    """
    def __init__(
        self,
        input_channels: int = 1,
        num_filters: int = 32,
        kernel_size: int = 3,
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
        # Note: the flatten + linear layer sizes assume at least one down‐sampling
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear((num_filters * 2) * 1, 64),  # adapt “1” if you know the exact temporal length
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: shape (batch, time) or (batch, 1, time)

        Returns:
            shape (batch,) probabilities in [0,1]
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = x.to(next(self.parameters()).device)
        features = self.features(x)
        return self.classifier(features).squeeze(-1)


def build_detection_model(cfg: dict) -> nn.Module:
    """
    Factory: instantiate CavitationDetectionModel on the best device,
    optionally JIT-compile on CUDA/CPU backends.
    """
    device = get_device()
    model = CavitationDetectionModel(
        input_channels=1,
        num_filters=cfg.get('det_filters', 32),
        kernel_size=cfg.get('det_kernel_size', 3),
    ).to(device)

    # TorchCompile yields speedups on CUDA/CPU
    if hasattr(torch, 'compile') and device.type != 'mps':
        model = torch.compile(model)

    return model
