"""Lightweight 2D Convolutional Neural Network for Log-Mel Spectrogram Audio Classification.

Optimized for fast, leak-free CPU execution on 64x128 Log-Mel spectrograms.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Convolutional building block with Conv2d, BatchNorm, ReLU, and MaxPool."""

    def __init__(self, in_channels: int, out_channels: int, pool: bool = True) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) if pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class CoughAudioCNN(nn.Module):
    """Lightweight 2D CNN baseline for COUGHVID Log-Mel Spectrograms.

    Architecture:
    - Input: (B, 1, 64, 128)
    - Block 1: Conv2d(1 -> 16, 3x3) + BN + ReLU + MaxPool(2, 2) -> (B, 16, 32, 64)
    - Block 2: Conv2d(16 -> 32, 3x3) + BN + ReLU + MaxPool(2, 2) -> (B, 32, 16, 32)
    - Block 3: Conv2d(32 -> 64, 3x3) + BN + ReLU + MaxPool(2, 2) -> (B, 64, 8, 16)
    - Block 4: Conv2d(64 -> 128, 3x3) + BN + ReLU + MaxPool(2, 2) -> (B, 128, 4, 8)
    - Global Average Pooling: AdaptiveAvgPool2d((1, 1)) -> (B, 128, 1, 1)
    - Classifier Head:
        Dropout(p=0.3)
        Linear(128 -> 32)
        ReLU
        Dropout(p=0.2)
        Linear(32 -> 1) -> Logits (B, 1)
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 16,
        fc_dim: int = 32,
        dropout_conv: float = 0.3,
        dropout_fc: float = 0.2,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels

        # Convolutional stages
        self.stage1 = ConvBlock(in_channels, base_channels, pool=True)        # -> 16 channels, (32, 64)
        self.stage2 = ConvBlock(base_channels, base_channels * 2, pool=True)  # -> 32 channels, (16, 32)
        self.stage3 = ConvBlock(base_channels * 2, base_channels * 4, pool=True)  # -> 64 channels, (8, 16)
        self.stage4 = ConvBlock(base_channels * 4, base_channels * 8, pool=True)  # -> 128 channels, (4, 8)

        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected classification head
        in_features = base_channels * 8
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_conv),
            nn.Linear(in_features, fc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_fc),
            nn.Linear(fc_dim, 1),
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Kaiming normal initialization for Conv layers, Xavier uniform for linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass computing scalar logits.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, 1, F, T) or (B, F, T).

        Returns
        -------
        torch.Tensor
            Logits of shape (B, 1).
        """
        if x.ndim == 3:
            x = x.unsqueeze(1)  # (B, 1, F, T)
        elif x.ndim != 4:
            raise ValueError(f"Expected 3D or 4D tensor, got shape {x.shape}")

        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected channel dimension {self.in_channels}, got {x.shape[1]} in shape {x.shape}"
            )

        feat = self.stage1(x)
        feat = self.stage2(feat)
        feat = self.stage3(feat)
        feat = self.stage4(feat)
        pooled = self.gap(feat)
        logits = self.classifier(pooled)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Compute sigmoid positive-class probabilities in [0.0, 1.0]."""
        logits = self.forward(x)
        return torch.sigmoid(logits)

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
