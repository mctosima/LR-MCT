"""Image-based LRCN-3Conv model from Aripin & Setiawan."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.pool(F.relu(self.conv(x))))


class LRCN3Conv(nn.Module):
    """Three per-frame convolution blocks followed by one LSTM."""

    def __init__(
        self,
        num_classes: int,
        input_channels: int = 1,
        input_size: tuple[int, int] = (80, 80),
    ) -> None:
        super().__init__()
        if num_classes <= 0 or input_channels <= 0:
            raise ValueError("num_classes and input_channels must be positive")
        if len(input_size) != 2 or min(input_size) <= 0 or input_size[0] % 8 or input_size[1] % 8:
            raise ValueError("input_size dimensions must be positive multiples of 8")
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.input_size = input_size
        self.blocks = nn.ModuleList(
            [
                _ConvBlock(input_channels, 16),
                _ConvBlock(16, 32),
                _ConvBlock(32, 64),
            ]
        )
        feature_size = 64 * (input_size[0] // 8) * (input_size[1] // 8)
        self.feature_size = feature_size
        self.lstm = nn.LSTM(feature_size, 64, num_layers=1, batch_first=True)
        self.dropout_lstm = nn.Dropout(0.5)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 5 or x.shape[1] != self.input_channels or tuple(x.shape[-2:]) != self.input_size:
            raise ValueError(
                f"Expected input shape (batch, {self.input_channels}, time, {self.input_size[0]}, {self.input_size[1]}), "
                f"got {tuple(x.shape)}"
            )
        batch, channels, steps, height, width = x.shape
        frames = x.reshape(batch * steps, channels, height, width)
        for block in self.blocks:
            frames = block(frames)
        sequence = frames.reshape(batch, steps, -1)
        output, _ = self.lstm(sequence)
        return self.fc(self.dropout_lstm(output[:, -1, :]))


def model_summary(model: LRCN3Conv, input_shape: tuple[int, int, int, int, int]) -> dict[str, Any]:
    """Return trainable count and approximate single-batch forward GFLOPs."""
    if len(input_shape) != 5:
        raise ValueError("input_shape must be (batch, channels, time, height, width)")
    batch, channels, steps, height, width = input_shape
    if channels != model.input_channels or (height, width) != model.input_size:
        raise ValueError("input_shape does not match model input")
    total_ops = 0
    current_h, current_w = height, width
    in_channels = channels
    for out_channels in (16, 32, 64):
        total_ops += batch * steps * current_h * current_w * (in_channels * 9 * out_channels * 2)
        current_h //= 2
        current_w //= 2
        total_ops += batch * steps * current_h * current_w * out_channels
        in_channels = out_channels
    feature_size = 64 * current_h * current_w
    total_ops += batch * steps * 8 * (feature_size * 64 + 64 * 64 + 64) * 2
    total_ops += batch * 64 * model.num_classes * 2
    return {
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "approx_gflops": total_ops / 1e9,
        "input_shape": input_shape,
    }
