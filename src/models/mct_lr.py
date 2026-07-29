"""MCT-LR — Multi-modal Cross-attention Transformer for Lip Reading.

Late-fusion architecture: each stream processes independently (proven
backbones from Utama and Aripin), then concatenated features pass through
a fusion head before classification.

Design:
- Image stream: Aripin's 3-conv blocks → flatten → LSTM → (B, 64)
- Landmark stream: Utama-inspired 2-layer BiLSTM + attention → (B, 128)
- Fusion: concat → Linear(192 → 128) → ReLU → Dropout → Linear(128 → K)

Why late fusion: avoids the 6400→128 compression bottleneck that killed
cross-attention gradient flow. Each branch is independently proven;
the only new component is the shallow fusion head.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class _ConvBlock(nn.Module):
    """Conv2d → BatchNorm → ReLU → MaxPool → Dropout (Aripin pattern)."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout(0.25)

    def forward(self, x: Tensor) -> Tensor:
        return self.drop(self.pool(F.relu(self.bn(self.conv(x)))))


# ---------------------------------------------------------------------------
# MCT-LR model (late fusion)
# ---------------------------------------------------------------------------

class MCTLR(nn.Module):
    """Late-fusion multimodal lip reading model.

    Args:
        num_classes: Number of output classes.
        img_size: Grayscale ROI size (must be multiple of 8).
    """

    def __init__(self, num_classes: int, img_size: tuple[int, int] = (80, 80)) -> None:
        super().__init__()
        if min(img_size) % 8 != 0:
            raise ValueError("img_size dimensions must be multiples of 8")
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")

        self.num_classes = num_classes
        self.img_size = img_size

        # ---- Image stream (Aripin backbone) --------------------------------
        self.img_conv1 = _ConvBlock(1, 16)
        self.img_conv2 = _ConvBlock(16, 32)
        self.img_conv3 = _ConvBlock(32, 64)
        feat_dim = 64 * (img_size[0] // 8) * (img_size[1] // 8)  # 6400
        self.img_lstm = nn.LSTM(feat_dim, 64, num_layers=1, batch_first=True)
        self.img_drop = nn.Dropout(0.5)

        # ---- Landmark stream (Utama-inspired) ------------------------------
        self.lm_lstm = nn.LSTM(
            input_size=80, hidden_size=64, num_layers=2,
            batch_first=True, bidirectional=True, dropout=0.5,
        )
        self.lm_attn = nn.Linear(128, 1)
        self.lm_drop = nn.Dropout(0.5)

        # ---- Fusion head ---------------------------------------------------
        self.fusion = nn.Sequential(
            nn.Linear(64 + 128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    # -------------------------------------------------------------------
    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # -------------------------------------------------------------------
    def forward(self, img: Tensor, lm: Tensor) -> Tensor:
        B, _, T_img, H, W = img.shape

        # ---- Image stream ----
        frames = img.permute(0, 2, 1, 3, 4).reshape(B * T_img, 1, H, W)
        frames = self.img_conv1(frames)
        frames = self.img_conv2(frames)
        frames = self.img_conv3(frames)                     # (B*T_img, 64, 10, 10)
        seq = frames.reshape(B, T_img, -1)                  # (B, T_img, 6400)
        _, (h, _) = self.img_lstm(seq)                      # h: (1, B, 64)
        img_feat = self.img_drop(h.squeeze(0))              # (B, 64)

        # ---- Landmark stream ----
        lm_seq, _ = self.lm_lstm(lm)                        # (B, T_lmk, 128)
        scores = self.lm_attn(lm_seq).squeeze(-1)
        weights = F.softmax(scores, dim=1)
        lm_feat = (lm_seq * weights.unsqueeze(-1)).sum(dim=1)  # (B, 128)
        lm_feat = self.lm_drop(lm_feat)

        # ---- Fuse and classify ----
        fused = torch.cat([img_feat, lm_feat], dim=-1)      # (B, 192)
        return self.fusion(fused)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def model_summary(
    model: MCTLR,
    img_shape: tuple[int, int, int, int, int],
    lm_shape: tuple[int, int, int],
) -> dict[str, Any]:
    total = model.n_params
    _, _, T_img, H, W = img_shape
    T_lmk = lm_shape[1]
    B = img_shape[0]

    def _flops(ci: int, co: int, k: int, hi: int, wi: int) -> float:
        return ci * co * k * k * hi * wi * 2

    ops = 0.0
    for ci, co in [(1, 16), (16, 32), (32, 64)]:
        ops += B * T_img * _flops(ci, co, 3, H, W)
        H //= 2
        W //= 2

    # LSTM image: 4 * (input+hidden) * hidden * 2
    ops += B * T_img * 4 * (6400 + 64) * 64 * 2
    # LSTM landmark (layer 1 + layer 2 bidirectional)
    for layer in range(2):
        inp = 80 if layer == 0 else 128
        hid = 64
        dirs = 2
        ops += B * T_lmk * dirs * 4 * (inp + hid) * hid * 2

    # fusion
    ops += B * (192 * 128 + 128 * model.num_classes) * 2
    return {
        "trainable_params": total,
        "approx_gflops": round(ops / 1e9, 4),
        "img_shape": img_shape,
        "lm_shape": lm_shape,
    }
