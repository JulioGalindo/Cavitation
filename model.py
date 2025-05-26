import torch
import torch.nn as nn
import torch.nn.functional as F


class CavitationModel(nn.Module):
    """
    Base model for cavitation detection and classification.
    Supports binary and multiclass classification.
    Optimized for Apple M1/M2/M4 (channels_last) and CUDA (Linux).
    """
    def __init__(self, input_length, num_classes=2, model_type="basic_cnn"):
        super().__init__()
        self.model_type = model_type
        self.num_classes = num_classes

        if model_type == "basic_cnn":
            self.model = nn.Sequential(
                nn.Conv1d(1, 16, kernel_size=5, stride=1, padding=2),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(32),
                nn.Flatten(),
                nn.Linear(32 * 32, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes)
            )

        elif model_type == "msc_cnn":
            self.model = MSCNN(input_length=input_length, num_classes=num_classes)

        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

    def forward(self, x):
        return self.model(x)


class MSCNN(nn.Module):
    """
    Multi-Scale CNN model adapted from cavitation detection literature.
    Input: (B, 1, L)
    """
    def __init__(self, input_length, num_classes):
        super().__init__()

        self.branch1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.branch2 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.branch3 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.combine = nn.Sequential(
            nn.Conv1d(48, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(32),
            nn.Flatten(),
            nn.Linear(64 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = torch.cat([x1, x2, x3], dim=1)
        return self.combine(x_cat)


def get_model(config):
    """
    Build model from config dict (YAML-parsed).
    """
    return CavitationModel(
        input_length=config["data_generator"]["sample_rate"] * config["data_generator"]["duration_seconds"],
        num_classes=config["model"]["num_classes"],
        model_type=config["model"].get("type", "basic_cnn")
    )
