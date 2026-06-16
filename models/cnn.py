"""CNN model for BreakHis breast cancer classification with configurable normalization.

Supports five normalization strategies selected via ``norm_type``:
    * ``"none"``     – no normalization layers (baseline)
    * ``"batch"``    – ``BatchNorm2d`` after each conv block
    * ``"group"``    – ``GroupNorm`` (8 groups) after each conv block
    * ``"layer"``    – ``LayerNorm`` (per-channel spatial) after each conv block
    * ``"instance"`` – ``InstanceNorm2d`` after each conv block
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn


VALID_NORM_TYPES = {"none", "batch", "group", "layer", "instance"}


def _make_norm_layer(norm_type: str, num_channels: int, spatial_size: int | None = None):
    """Factory for normalization layers.

    Parameters
    ----------
    norm_type : str
        One of ``VALID_NORM_TYPES``.
    num_channels : int
        Number of feature-map channels.
    spatial_size : int or None
        Spatial dimension (H=W) of the feature map.  Required for ``"layer"``
        normalization which needs the full normalized shape.
    """
    if norm_type == "none":
        return nn.Identity()
    if norm_type == "batch":
        return nn.BatchNorm2d(num_channels)
    if norm_type == "group":
        # Pick a group count that evenly divides num_channels
        num_groups = min(8, num_channels)
        while num_channels % num_groups != 0:
            num_groups -= 1
        return nn.GroupNorm(num_groups, num_channels)
    if norm_type == "layer":
        if spatial_size is None:
            raise ValueError("spatial_size is required for LayerNorm")
        return nn.LayerNorm([num_channels, spatial_size, spatial_size])
    if norm_type == "instance":
        return nn.InstanceNorm2d(num_channels, affine=True)

    raise ValueError(f"Unknown norm_type: {norm_type!r}. Must be one of {VALID_NORM_TYPES}")


class ConvBlock(nn.Module):
    """Conv → Norm → ReLU → Conv → Norm → ReLU → MaxPool."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_type: str = "batch",
        spatial_size: int | None = None,
        pool: bool = True,
    ):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=(norm_type == "none")),
            _make_norm_layer(norm_type, out_channels, spatial_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=(norm_type == "none")),
            _make_norm_layer(norm_type, out_channels, spatial_size),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BreakHisCNN(nn.Module):
    """4-block CNN for BreakHis binary classification.

    Input: (B, 3, 128, 128)
    Output: (B, num_classes)

    Architecture
    ------------
    Block 1: 3  → 32  channels, pool → 64×64
    Block 2: 32 → 64  channels, pool → 32×32
    Block 3: 64 → 128 channels, pool → 16×16
    Block 4: 128→ 256 channels, pool → 8×8
    AdaptiveAvgPool → (256,)
    Dropout → Linear → num_classes
    """

    def __init__(self, num_classes: int = 2, norm_type: str = "batch", dropout: float = 0.3):
        super().__init__()
        if norm_type not in VALID_NORM_TYPES:
            raise ValueError(f"norm_type must be one of {VALID_NORM_TYPES}, got {norm_type!r}")

        self.norm_type = norm_type
        self.num_classes = num_classes

        # Spatial sizes after each pool: 128→64→32→16→8
        self.features = nn.Sequential(
            ConvBlock(3, 32, norm_type, spatial_size=128),    # → 64
            nn.Dropout2d(dropout / 3),
            ConvBlock(32, 64, norm_type, spatial_size=64),    # → 32
            nn.Dropout2d(dropout / 2),
            ConvBlock(64, 128, norm_type, spatial_size=32),   # → 16
            nn.Dropout2d(dropout / 2),
            ConvBlock(128, 256, norm_type, spatial_size=16),  # → 8
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

    def reset_bn_stats(self):
        """Reset all BatchNorm running statistics to defaults.

        Must be called after FedAvg aggregation for models that use
        BatchNorm, because averaging running statistics from non-IID
        clients produces meaningless values.
        """
        for module in self.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                module.running_mean.zero_()
                module.running_var.fill_(1.0)
                module.num_batches_tracked.zero_()

    @torch.no_grad()
    def calibrate_bn(self, data_loaders, device, max_batches=10):
        """Recalibrate BatchNorm running statistics using representative data.

        After FedAvg aggregation, run a forward pass through a small sample
        from each client to compute globally valid running statistics.

        Parameters
        ----------
        data_loaders : list[DataLoader]
            Client training data loaders.
        device : torch.device
            Device to run calibration on.
        max_batches : int
            Maximum batches per client to use for calibration.
        """
        # Only calibrate if model actually has BatchNorm layers
        has_bn = any(
            isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d))
            for m in self.modules()
        )
        if not has_bn:
            return

        self.reset_bn_stats()
        self.train()  # BN must be in train mode to update running stats

        for loader in data_loaders:
            batch_count = 0
            for inputs, _ in loader:
                inputs = inputs.to(device)
                self(inputs)  # forward pass updates BN running stats
                batch_count += 1
                if batch_count >= max_batches:
                    break

        self.eval()  # return to eval mode

    def clone(self) -> "BreakHisCNN":
        """Return a detached deep copy for local client training."""
        return copy.deepcopy(self)


# ---------------------------------------------------------------------------
# Federated aggregation utilities
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClientUpdate:
    """A single client's trained state and accounting information."""

    client_id: int
    state_dict: dict[str, torch.Tensor]
    num_samples: int
    train_loss: float
    train_accuracy: float


def fedavg(client_updates: Sequence[ClientUpdate]) -> dict[str, torch.Tensor]:
    """Aggregate client model states using sample-weighted FedAvg.

    Averages all floating-point parameters and buffers (weights, biases,
    running_mean, running_var).  Integer buffers like ``num_batches_tracked``
    are reset to zero so that subsequent BN calibration starts fresh.
    """
    if not client_updates:
        raise ValueError("fedavg requires at least one client update.")

    total_samples = sum(update.num_samples for update in client_updates)
    if total_samples <= 0:
        raise ValueError("fedavg requires at least one trained sample.")

    averaged_state = copy.deepcopy(client_updates[0].state_dict)

    for key in averaged_state:
        first_value = averaged_state[key]
        if torch.is_floating_point(first_value):
            averaged_state[key] = sum(
                update.state_dict[key].float() * (update.num_samples / total_samples)
                for update in client_updates
            ).to(first_value.dtype)
        else:
            # Integer buffers (e.g. num_batches_tracked): reset to zero.
            # They will be recomputed during BN calibration.
            averaged_state[key] = torch.zeros_like(first_value)

    return averaged_state
