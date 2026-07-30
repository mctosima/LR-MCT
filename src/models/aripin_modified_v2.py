"""AripinModifiedV2 — two-stream image-only visual speech recognition.

Raw appearance path retains Aripin's three ConvBlock → LSTM design.
Signed motion path (positive/negative frame differences) uses a compact
separate encoder. Streams fuse through a zero-initialized residual gate.
Tail padding is masked via packed sequences and masked attention pooling.

Reference:
  Petridis et al.  End-to-End Visual Speech Recognition for Small-Scale
  Datasets.  Pattern Recognition Letters, 2019.  arXiv:1904.01954
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence


_ATTN_MASK_VALUE = -1e9


class _ConvBlock(nn.Module):
    """Single spatial block: conv → ReLU → 2×2 MaxPool → dropout."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.25) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.pool(F.relu(self.conv(x))))


class AripinModifiedV2(nn.Module):
    """Two-stream image model with packed-sequence temporal masking."""

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

        # --- raw appearance stream (identical to LRCN3Conv spatial backbone) ---
        self.raw_blocks = nn.ModuleList(
            [
                _ConvBlock(input_channels, 16),
                _ConvBlock(16, 32),
                _ConvBlock(32, 64),
            ]
        )
        raw_feat_dim = 64 * (input_size[0] // 8) * (input_size[1] // 8)  # 64×10×10 = 6400
        self.raw_lstm = nn.LSTM(raw_feat_dim, 64, num_layers=1, batch_first=True)
        self.raw_dropout = nn.Dropout(0.5)

        # --- motion stream (signed difference) ---
        self.motion_blocks = nn.ModuleList(
            [
                _ConvBlock(2, 8),
                _ConvBlock(8, 16),
                _ConvBlock(16, 32),
            ]
        )
        motion_feat_dim = 32 * (input_size[0] // 8) * (input_size[1] // 8)  # 32×10×10 = 3200
        self.motion_lstm = nn.LSTM(motion_feat_dim, 64, num_layers=1, batch_first=True)
        self.motion_dropout = nn.Dropout(0.5)

        # --- residual fusion gate ---
        self.gate = nn.Linear(128, 64)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)

        # --- masked additive attention ---
        self.attn_score: nn.Module = nn.Sequential(
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1, bias=False),
        )

        # --- classifier ---
        self.classifier = nn.Sequential(
            nn.LayerNorm(128),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _valid_lengths(x: Tensor) -> Tensor:
        """Return valid temporal length per sample (last non-zero frame index + 1).

        Expects x shaped (batch, channels, time, height, width).
        Raises ValueError if any sample is entirely black.
        """
        mask = x.abs().sum(dim=(1, 3, 4)) > 0.0  # (B, T)
        if not mask.any(dim=1).all():
            raise ValueError("Each sample must contain at least one non-zero ROI frame")
        lengths = mask.long().argmax(dim=1)  # first True index
        # scan from end to find last True
        rev = mask.flip(1)
        last_idx = mask.shape[1] - 1 - rev.long().argmax(dim=1)
        return last_idx + 1

    @staticmethod
    def _per_frame_spatial(
        x: Tensor,
        blocks: nn.ModuleList,
    ) -> Tensor:
        """Apply per-frame spatial blocks: (B*T, C, H, W) → (B*T, D)."""
        batch_t, c, h, w = x.shape
        frames = x
        for block in blocks:
            frames = block(frames)
        return frames.reshape(batch_t, -1)

    def _lstm_encode(
        self,
        lstm: nn.LSTM,
        dropout: nn.Dropout,
        features: Tensor,
        lengths: Tensor,
        batch: int,
        max_len: int,
    ) -> Tensor:
        """Pack → LSTM → pad → dropout → (B, max_len, 64)."""
        sorted_len, sort_idx = lengths.sort(descending=True)
        sorted_feat = features[sort_idx]
        packed = pack_padded_sequence(sorted_feat, sorted_len.cpu(), batch_first=True, enforce_sorted=True)
        packed_out, _ = lstm(packed)
        padded_out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=max_len)
        # restore original order
        rev_idx = torch.empty_like(sort_idx)
        rev_idx[sort_idx] = torch.arange(batch, device=sort_idx.device)
        return dropout(padded_out[rev_idx])

    def _masked_attention(
        self,
        fused: Tensor,
        lengths: Tensor,
    ) -> Tensor:
        """Masked additive attention over (B, T, 64).  Returns (B, 64)."""
        scores: Tensor = self.attn_score(fused).squeeze(-1)  # (B, T)
        arange = torch.arange(fused.shape[1], device=fused.device).unsqueeze(0)
        mask = arange >= lengths.unsqueeze(1)
        scores = scores.masked_fill(mask, _ATTN_MASK_VALUE)
        weights = F.softmax(scores, dim=1).unsqueeze(-1)
        return (fused * weights).sum(dim=1)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 5 or x.shape[1] != self.input_channels or tuple(x.shape[-2:]) != self.input_size:
            raise ValueError(
                f"Expected (batch, {self.input_channels}, time, {self.input_size[0]}, {self.input_size[1]}), "
                f"got {tuple(x.shape)}"
            )
        batch, channels, steps, height, width = x.shape
        if batch <= 0 or steps <= 0:
            raise ValueError("batch and time dimensions must be positive")

        lengths = self._valid_lengths(x)  # (B,)

        # --- raw appearance stream ---
        raw_frames = x.reshape(batch * steps, channels, height, width)
        raw_feat = self._per_frame_spatial(raw_frames, self.raw_blocks)  # (B*T, 6400)
        raw_feat = raw_feat.reshape(batch, steps, -1)
        raw_seq = self._lstm_encode(self.raw_lstm, self.raw_dropout, raw_feat, lengths, batch, steps)

        # --- motion stream ---
        # compute consecutive signed differences; split into +/- channels
        delta_full = torch.zeros(batch, channels, steps, height, width, device=x.device, dtype=x.dtype)
        delta_full[:, :, 1:] = x[:, :, 1:, :, :] - x[:, :, :-1, :, :]
        delta = delta_full
        # zero-out padding frames
        for b in range(batch):
            l = int(lengths[b].item())
            if l < steps:
                delta[b, :, l:, :, :] = 0.0

        pos = torch.clamp(delta, min=0.0)
        neg = torch.clamp(-delta, min=0.0)
        motion_input = torch.cat((pos, neg), dim=1)  # (B, 2, T, H, W)
        motion_frames = motion_input.permute(0, 2, 1, 3, 4).reshape(batch * steps, 2, height, width)
        motion_feat = self._per_frame_spatial(motion_frames, self.motion_blocks)  # (B*T, 3200)
        motion_feat = motion_feat.reshape(batch, steps, -1)
        motion_seq = self._lstm_encode(self.motion_lstm, self.motion_dropout, motion_feat, lengths, batch, steps)

        # --- fusion ---
        concat = torch.cat((raw_seq, motion_seq), dim=2)  # (B, T, 128)
        gate = torch.sigmoid(self.gate(concat))  # (B, T, 64)
        fused = raw_seq + gate * motion_seq  # (B, T, 64)

        # --- sequence summaries ---
        # last valid timestep per sample
        last_idx = (lengths - 1).unsqueeze(1).unsqueeze(2).expand(-1, 1, 64)
        last_state = fused.gather(1, last_idx).squeeze(1)  # (B, 64)

        # masked additive attention
        attn_vec = self._masked_attention(fused, lengths)  # (B, 64)

        # --- classifier ---
        combined = torch.cat((last_state, attn_vec), dim=1)  # (B, 128)
        return self.classifier(combined)


# ------------------------------------------------------------------
# model summary
# ------------------------------------------------------------------


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


def model_summary(
    model: AripinModifiedV2,
    input_shape: tuple[int, int, int, int, int],
) -> dict[str, Any]:
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

    # raw spatial stream
    for out_c in (16, 32, 64):
        total_ops += _conv_ops(batch, steps, channels, out_c, current_h, current_w, 3)
        current_h //= 2
        current_w //= 2
        total_ops += batch * steps * current_h * current_w * out_c  # ReLU + MaxPool approx
        channels = out_c

    # raw LSTM  (6400 -> 64)
    total_ops += batch * steps * 8 * (6400 * 64 + 64 * 64 + 64) * 2

    # motion spatial stream
    current_h, current_w = height, height
    for out_c in (8, 16, 32):
        total_ops += _conv_ops(batch, steps, 2, out_c, current_h, current_w, 3)
        current_h //= 2
        current_w //= 2
        total_ops += batch * steps * current_h * current_w * out_c

    # motion LSTM (3200 -> 64)
    total_ops += batch * steps * 8 * (3200 * 64 + 64 * 64 + 64) * 2

    # gate, attention, classifier (tiny)
    total_ops += 2 * batch * 128 * 64  # gate Linear
    total_ops += 2 * batch * (64 * 32 + 32)  # attention
    total_ops += 2 * batch * (128 * model.num_classes)  # classifier

    return {
        "trainable_params": model.n_params,
        "approx_gflops": total_ops / 1e9,
        "input_shape": input_shape,
    }
