"""ConvNeXt-inspired image LRCN model for Aripin comparison."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ConvNeXtBlock(nn.Module):
    """Residual ConvNeXt block using depthwise spatial mixing and an MLP."""

    def __init__(self, dim: int, expand_ratio: int = 4) -> None:
        super().__init__()
        if dim <= 0 or expand_ratio <= 0:
            raise ValueError("dim and expand_ratio must be positive")
        self.dim = dim
        self.expand_ratio = expand_ratio
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, dim * expand_ratio)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(dim * expand_ratio, dim)
        self.gamma = nn.Parameter(torch.ones(dim) * 1e-6)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or x.shape[1] != self.dim:
            raise ValueError(f"Expected (batch, {self.dim}, height, width), got {tuple(x.shape)}")
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x * self.gamma
        x = x.permute(0, 3, 1, 2)
        return x + residual


class _Downsample(nn.Module):
    """ConvNeXt downsampling layer with channels-last LayerNorm."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        if in_dim <= 0 or out_dim <= 0:
            raise ValueError("downsample dimensions must be positive")
        self.in_dim = in_dim
        self.norm = nn.LayerNorm(in_dim)
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or x.shape[1] != self.in_dim:
            raise ValueError(f"Expected (batch, {self.in_dim}, height, width), got {tuple(x.shape)}")
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return self.conv(x)


class AripinModified(nn.Module):
    """ConvNeXt-inspired per-frame CNN followed by a bidirectional LSTM."""

    def __init__(
        self,
        num_classes: int,
        input_channels: int = 1,
        input_size: tuple[int, int] = (80, 80),
    ) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if input_channels <= 0:
            raise ValueError("input_channels must be positive")
        if len(input_size) != 2 or min(input_size) <= 0 or any(size % 16 for size in input_size):
            raise ValueError("input_size dimensions must be positive multiples of 16")
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.input_size = input_size

        self.stem = nn.Conv2d(input_channels, 64, kernel_size=4, stride=4)
        self.downsample1 = _Downsample(64, 128)
        self.stage1 = nn.Sequential(*(ConvNeXtBlock(128) for _ in range(3)))
        self.downsample2 = _Downsample(128, 256)
        self.stage2 = nn.Sequential(*(ConvNeXtBlock(256) for _ in range(3)))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=192,
            num_layers=2,
            dropout=0.3,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(384),
            nn.Linear(384, 192),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(192, num_classes),
        )

    @property
    def n_params(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 5:
            raise ValueError(
                "Expected input shape "
                f"(batch, {self.input_channels}, time, {self.input_size[0]}, {self.input_size[1]}), "
                f"got {tuple(x.shape)}"
            )
        if x.shape[1] != self.input_channels or tuple(x.shape[-2:]) != self.input_size:
            raise ValueError(
                "Expected input shape "
                f"(batch, {self.input_channels}, time, {self.input_size[0]}, {self.input_size[1]}), "
                f"got {tuple(x.shape)}"
            )
        batch, channels, steps, height, width = x.shape
        if batch <= 0 or steps <= 0:
            raise ValueError("batch and time dimensions must be positive")
        frames = x.reshape(batch * steps, channels, height, width)
        frames = self.stem(frames)
        frames = self.downsample1(frames)
        frames = self.stage1(frames)
        frames = self.downsample2(frames)
        frames = self.stage2(frames)
        frames = self.pool(frames).flatten(1)
        sequence = frames.reshape(batch, steps, 256)
        output, _ = self.lstm(sequence)
        forward_final = output[:, -1, :192]
        backward_initial = output[:, 0, 192:]
        return self.classifier(torch.cat((forward_final, backward_initial), dim=1))


def _conv_ops(
    batch: int,
    steps: int,
    in_channels: int,
    out_channels: int,
    height: int,
    width: int,
    kernel: int,
    groups: int = 1,
) -> int:
    return 2 * batch * steps * out_channels * height * width * (in_channels // groups) * kernel * kernel


def model_summary(model: AripinModified, input_shape: tuple[int, int, int, int, int]) -> dict[str, Any]:
    """Return trainable parameter count and approximate forward GFLOPs."""
    if len(input_shape) != 5:
        raise ValueError("input_shape must be (batch, channels, time, height, width)")
    batch, channels, steps, height, width = input_shape
    if batch <= 0 or steps <= 0:
        raise ValueError("batch and time dimensions must be positive")
    if channels != model.input_channels or (height, width) != model.input_size:
        raise ValueError("input_shape does not match model input")

    total_ops = 0
    current_h, current_w = height, width
    total_ops += _conv_ops(batch, steps, channels, 64, current_h // 4, current_w // 4, 4)
    current_h //= 4
    current_w //= 4
    total_ops += _conv_ops(batch, steps, 64, 128, current_h // 2, current_w // 2, 2)
    current_h //= 2
    current_w //= 2
    for _ in range(3):
        total_ops += _conv_ops(batch, steps, 128, 128, current_h, current_w, 7, groups=128)
        total_ops += 2 * batch * steps * current_h * current_w * (128 * 512 + 512 * 128)
    total_ops += _conv_ops(batch, steps, 128, 256, current_h // 2, current_w // 2, 2)
    current_h //= 2
    current_w //= 2
    for _ in range(3):
        total_ops += _conv_ops(batch, steps, 256, 256, current_h, current_w, 7, groups=256)
        total_ops += 2 * batch * steps * current_h * current_w * (256 * 1024 + 1024 * 256)
    total_ops += 2 * batch * steps * 2 * 4 * 192 * (256 + 192 + 1) * 2
    total_ops += 2 * batch * (384 * 192 + 192 * model.num_classes)
    return {
        "trainable_params": model.n_params,
        "approx_gflops": total_ops / 1e9,
        "input_shape": input_shape,
    }
