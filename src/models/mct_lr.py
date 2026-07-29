"""MCT-LR — Multi-modal Cross-attention Transformer for Lip Reading.

Fuses per-frame grayscale mouth ROI (image stream) with MediaPipe lip
landmarks (geometric stream) via cross-attention, followed by a BiGRU
temporal backend with learned attention pooling.

Architecture draws from:
- LipFormer (IEEE TCSVT 2023): visual-landmark transformer fusion
- Cross-Attention Fusion (Daou et al. 2024): ResNet + cross-attn + MS-TCN

Design decisions for small-dataset / small-batch regime:
- Per-frame 2D convs (no 3D conv — too many params, unstable at batch=4)
- LayerNorm throughout (stable at any batch size, unlike BatchNorm)
- Single-direction cross-attention (image Q attends to landmark KV)
- No position embedding on image stream (conv already carries spatial info)
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def _conv_block(in_ch: int, out_ch: int, stride: int = 1, groups: int = 1) -> nn.Module:
    """Conv2d + LayerNorm + GELU, optionally strided."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False, groups=groups),
        nn.GroupNorm(num_groups=max(1, out_ch // 8) if out_ch >= 8 else 1, num_channels=out_ch),
        nn.ReLU(inplace=True),
    )


class _ConvStage(nn.Module):
    """Stack of conv blocks; first block optionally changes stride."""

    def __init__(self, in_ch: int, out_ch: int, blocks: int, stride: int = 1) -> None:
        super().__init__()
        layers = [_conv_block(in_ch, out_ch, stride)]
        for _ in range(1, blocks):
            layers.append(_conv_block(out_ch, out_ch))
        self.stage = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.stage(x)


class _PositionalEncoding(nn.Module):
    """Sinusoidal PE for ``batch_first`` tensors."""

    def __init__(self, d_model: int, max_len: int = 500) -> None:
        super().__init__()
        pe = torch.zeros(1, max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[0, :, 0::2] = torch.sin(pos * div)
        pe[0, :, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pe[:, : x.size(1)]


# ---------------------------------------------------------------------------
# MCT-LR model
# ---------------------------------------------------------------------------

class MCTLR(nn.Module):
    """Multi-modal cross-attention transformer for lip reading.

    Args:
        num_classes: Number of output word/phrase classes.
        img_size: Spatial size of grayscale ROI (default ``(80, 80)``).
    """

    def __init__(self, num_classes: int, img_size: tuple[int, int] = (80, 80)) -> None:
        super().__init__()
        if min(img_size) % 8 != 0:
            raise ValueError("img_size dimensions must be multiples of 8")
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")

        self.num_classes = num_classes
        self.img_size = img_size

        # ---- Image stream (per-frame 2D convs, stable at batch=4) ----------
        # (B, T_img, 1, 80, 80) reshaped to (B*T_img, 1, 80, 80)
        self.img_stem = nn.Sequential(
            nn.Conv2d(1, 32, 7, 2, 3, bias=False),
            nn.GroupNorm(4, 32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),   # → (B*T, 32, 20, 20)
        )
        self.img_stage1 = _ConvStage(32, 64, 2)             # → (B*T, 64, 20, 20)
        self.img_stage2 = _ConvStage(64, 128, 2, stride=2) # → (B*T, 128, 10, 10)
        self.img_stage3 = _ConvStage(128, 256, 2, stride=2) # → (B*T, 256, 5, 5)
        self.img_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.img_out_dim = 256

        # ---- Landmark stream ------------------------------------------------
        self.lm_proj = nn.Sequential(
            nn.Linear(80, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        self.lm_pe = _PositionalEncoding(128)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=4, dim_feedforward=512, dropout=0.1, batch_first=True,
        )
        self.lm_transformer = nn.TransformerEncoder(enc_layer, num_layers=2)

        # ---- Cross-attention fusion (image attends to landmarks) ------------
        self.xattn = nn.MultiheadAttention(embed_dim=128, num_heads=4, dropout=0.1, batch_first=True)
        self.img_to_q = nn.Linear(256, 128)
        self.lm_to_kv = nn.Linear(128, 128)
        self.fuse_linear = nn.Sequential(
            nn.Linear(128 + 128, 256),  # cross-attn output(128) + landmark global(128)
            nn.LayerNorm(256),
        )

        # ---- Temporal backend -----------------------------------------------
        self.bigru = nn.GRU(
            input_size=256, hidden_size=128, num_layers=2,
            dropout=0.3, batch_first=True, bidirectional=True,
        )
        self.attn = nn.Linear(256, 1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256, num_classes)

    # -------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # -------------------------------------------------------------------
    # Forward
    # -------------------------------------------------------------------

    def forward(self, img: Tensor, lm: Tensor) -> Tensor:
        """Forward pass.

        Args:
            img: ``(B, 1, T_img, H, W)``.
            lm:  ``(B, T_lmk, 80)``.

        Returns:
            ``(B, num_classes)`` logits.
        """
        B, _, T_img, H, W = img.shape
        T_lmk = lm.shape[1]

        # ---- Image stream (per-frame) ----
        frames = img.permute(0, 2, 1, 3, 4).reshape(B * T_img, 1, H, W)
        x = self.img_stem(frames)
        x = self.img_stage1(x)
        x = self.img_stage2(x)
        x = self.img_stage3(x)
        x = self.img_pool(x).view(B, T_img, self.img_out_dim)  # (B, T_img, 256)

        # ---- Landmark stream ----
        lm_feat = self.lm_proj(lm)                            # (B, T_lmk, 128)
        lm_feat = self.lm_pe(lm_feat)
        lm_feat = self.lm_transformer(lm_feat)                # (B, T_lmk, 128)

        # ---- Cross-attention: image attends to landmarks ----
        Q = self.img_to_q(x)                                   # (B, T_img, 128)
        KV = self.lm_to_kv(lm_feat)                            # (B, T_lmk, 128)
        attended, _ = self.xattn(Q, KV, KV)                    # (B, T_img, 128)

        # Global landmark feature (mean pool → broadcast)
        lm_global = lm_feat.mean(dim=1, keepdim=True)          # (B, 1, 128)
        lm_global = lm_global.expand(-1, T_img, -1)            # (B, T_img, 128)

        fused = torch.cat([attended, lm_global], dim=-1)       # (B, T_img, 256)
        fused = self.fuse_linear(fused)                        # (B, T_img, 256)

        # ---- Temporal backend ----
        seq, _ = self.bigru(fused)                             # (B, T_img, 256)
        scores = self.attn(seq).squeeze(-1)                    # (B, T_img)
        weights = F.softmax(scores, dim=1)
        context = (seq * weights.unsqueeze(-1)).sum(dim=1)     # (B, 256)
        return self.fc(self.dropout(context))


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def model_summary(
    model: MCTLR,
    img_shape: tuple[int, int, int, int, int],
    lm_shape: tuple[int, int, int],
) -> dict[str, Any]:
    """Return param count and estimated GFLOPs."""
    _, _, T_img, H, W = img_shape
    T_lmk = lm_shape[1]
    total = model.n_params
    B = img_shape[0]

    def _flops_conv2d(ci: int, co: int, k: int, hi: int, wi: int, stride: int) -> float:
        ho = (hi - k + 2 * (k // 2)) // stride + 1
        wo = (wi - k + 2 * (k // 2)) // stride + 1
        return ci * co * k * k * ho * wo * 2  # mul+add per op

    ops = 0.0
    # stem conv → (40,40)
    ops += B * T_img * _flops_conv2d(1, 32, 7, 80, 80, 2)
    sp, ch = 40, 32
    # maxpool: negligible
    sp = 20
    # stages
    for (_in, _out, _blk, _str) in [(32, 64, 2, 1), (64, 128, 2, 2), (128, 256, 2, 2)]:
        if _str == 2:
            sp //= 2
        for _ in range(_blk):
            ops += B * T_img * _flops_conv2d(_in, _out, 3, sp, sp, 1)
            _in = _out

    # landmark transformer (rough)
    d_lm, ff = 128, 512
    for _ in range(2):
        ops += B * (4 * T_lmk * d_lm * d_lm + 2 * T_lmk * T_lmk * d_lm)
        ops += B * 2 * T_lmk * d_lm * ff * 2

    # cross-attention
    ops += B * (3 * T_img * 128 * 128 + T_img * T_lmk * 128)

    # BiGRU
    for lidx in range(2):
        inp = 256 if lidx == 0 else 256
        ops += B * T_img * 2 * 3 * (inp + 128) * 128 * 2

    ops += B * 256 * model.num_classes * 2  # FC

    return {
        "trainable_params": total,
        "approx_gflops": round(ops / 1e9, 4),
        "img_shape": img_shape,
        "lm_shape": lm_shape,
    }
