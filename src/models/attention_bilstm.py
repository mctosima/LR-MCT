"""Attention-BiLSTM model used for the Utama et al. baseline."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class AttentionBiLSTM(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_size: int = 64, dropout: float = 0.5):
        super().__init__()
        if input_dim <= 0 or num_classes <= 0 or hidden_size <= 0:
            raise ValueError("input_dim, num_classes, and hidden_size must be positive")
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.attn = nn.Linear(hidden_size * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3 or x.shape[-1] != self.input_dim:
            raise ValueError(f"Expected input shape (batch, time, {self.input_dim}), got {tuple(x.shape)}")
        sequence, _ = self.lstm(x)
        scores = self.attn(sequence).squeeze(-1)
        weights = F.softmax(scores, dim=1)
        context = (sequence * weights.unsqueeze(-1)).sum(dim=1)
        return self.fc(self.dropout(context))


def model_summary(model: AttentionBiLSTM, input_shape: tuple[int, int, int]) -> dict[str, Any]:
    """Return trainable parameter count and approximate single-batch GFLOPs."""
    if len(input_shape) != 3 or input_shape[2] != model.input_dim:
        raise ValueError(f"input_shape must be (batch, time, {model.input_dim})")
    batch, steps, _ = input_shape
    directions = 2
    hidden = model.hidden_size
    layer1_ops = batch * steps * directions * 2 * 4 * hidden * (model.input_dim + hidden + 1)
    layer2_ops = batch * steps * directions * 2 * 4 * hidden * (directions * hidden + hidden + 1)
    lstm_ops = layer1_ops + layer2_ops
    attention_ops = batch * steps * ((2 * hidden) * 2 + 2 * hidden * 2)
    classifier_ops = batch * (2 * hidden) * model.num_classes * 2
    return {
        "trainable_params": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "approx_gflops": (lstm_ops + attention_ops + classifier_ops) / 1e9,
        "input_shape": input_shape,
    }
