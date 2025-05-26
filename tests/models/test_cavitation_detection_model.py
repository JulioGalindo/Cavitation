# models/cavitation_detection_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

def build_detection_model(cfg: dict):
    """
    Build a cavitation detection model.

    cfg must contain:
      - sample_rate, duration_seconds              # to infer input length if needed
      - det_filters, det_kernel_size               # for feature extractor
      - time_collapse (bool): if True, output one probability per example;
                              if False, output a probability per time step
    """
    return CavitationDetectionModel(
        in_length=int(cfg['sample_rate'] * cfg['duration_seconds']),
        filters=cfg['det_filters'],
        kernel_size=cfg['det_kernel_size'],
        time_collapse=cfg.get('time_collapse', True)
    )

class CavitationDetectionModel(nn.Module):
    """
    A simple 1D CNN for binary cavitation detection, with optional time collapse.
    """
    def __init__(self, in_length, filters, kernel_size, time_collapse: bool = True):
        super().__init__()
        self.time_collapse = time_collapse

        # Feature extractor: input [B, L] → [B, C, L']
        self.features = nn.Sequential(
            nn.Conv1d(1, filters, kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.Conv1d(filters, filters, kernel_size, padding=kernel_size//2),
            nn.ReLU(),
        )

        if self.time_collapse:
            # collapse time dimension with global average pooling
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),      # → [B, C, 1]
                nn.Flatten(1),                # → [B, C]
                nn.Linear(filters, 1),        # → [B, 1]
                nn.Sigmoid()
            )
        else:
            # keep full time resolution: classification per time step
            # features output has shape [B, C, L']
            self.classifier = nn.Sequential(
                nn.Conv1d(filters, 1, kernel_size=1),  # → [B, 1, L']
                nn.Sigmoid()
            )

    def forward(self, x: torch.Tensor):
        """
        x: [B, L] or [B, 1, L]
        returns:
          if time_collapse, shape [B] of probabilities;
          else, shape [B, L'] of probabilities per time step.
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, 1, L]
        feats = self.features(x)  # [B, C, L']
        out = self.classifier(feats)
        if self.time_collapse:
            return out.squeeze(1)  # [B]
        else:
            return out.squeeze(1)  # [B, L']

