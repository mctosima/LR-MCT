"""MCT-LR — Multi-modal Cross-attention Transformer for Lip Reading.

Fuses grayscale mouth ROI (image stream) with MediaPipe lip landmarks
(geometric stream) via bidirectional cross-attention, followed by a
BiGRU temporal backend with learned attention pooling.

Architecture draws from:
- LipFormer (IEEE TCSVT 2023): visual-landmark transformer fusion
- Cross-Attention Fusion (Daou et al. 2024): ResNet + cross-attn + MS-TCN
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

class _ResBlock(nn.Module):
    """Two 3×3 convs with optional stride and 1×1 shortcut."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class _ResStage(nn.Module):
    """Stack of _ResBlocks; first block changes channels + optional stride."""

    def __init__(self, in_channels: int, out_channels: int, blocks: int, stride: int = 1) -> None:
        super().__init__()
        layers = [_ResBlock(in_channels, out_channels, stride)]
        for _ in range(1, blocks):
            layers.append(_ResBlock(out_channels, out_channels))
        self.stage = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.stage(x)


class _PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding matching ``batch_first`` convention."""

    def __init__(self, d_model: int, max_len: int = 500) -> None:
        super().__init__()
        pe = torch.zeros(1, max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
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

        # ---- Image stream ---------------------------------------------------
        self.stem3d = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
        )
        # After stem3d: (B, 32, T, H/4, W/4) = (B, 32, T, 20, 20)

        self.img_stage1 = _ResStage(32, 64, blocks=2, stride=1)   # → (B, 64, T, 20, 20)
        self.img_stage2 = _ResStage(64, 128, blocks=2, stride=2)  # → (B, 128, T, 10, 10)
        self.img_stage3 = _ResStage(128, 256, blocks=2, stride=2) # → (B, 256, T, 5, 5)
        self.img_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.img_out_dim = 256

        # ---- Landmark stream ------------------------------------------------
        self.lm_embed = nn.Sequential(
            nn.Linear(80, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        self.lm_pe = _PositionalEncoding(128)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=4, dim_feedforward=512, dropout=0.1, batch_first=True,
        )
        self.lm_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.lm_out_dim = 128

        # ---- Fusion (bidirectional cross-attention) -------------------------
        self.xattn_img_to_lmk = nn.MultiheadAttention(embed_dim=128, num_heads=4, dropout=0.1, batch_first=True)
        self.xattn_lmk_to_img = nn.MultiheadAttention(embed_dim=128, num_heads=4, dropout=0.1, batch_first=True)

        self.proj_img_q = nn.Linear(256, 128)   # image → 128-dim query
        self.proj_lmk_kv = nn.Linear(128, 128)  # landmark → 128-dim KV
        self.proj_img_fused = nn.Linear(128, 256)

        self.proj_lmk_q = nn.Linear(128, 128)
        self.proj_img_kv = nn.Linear(256, 128)
        self.proj_lmk_aligned = nn.Linear(128, 128)

        self.fusion_linear = nn.Sequential(
            nn.Linear(256 + 128, 256),
            nn.LayerNorm(256),
        )

        # ---- Temporal backend (BiGRU + attention) ---------------------------
        self.bigru = nn.GRU(
            input_size=256, hidden_size=128, num_layers=2,
            dropout=0.3, batch_first=True, bidirectional=True,
        )
        self.attn = nn.Linear(256, 1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256, num_classes)

    # -------------------------------------------------------------------
    # Public validation helpers
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
            img: ``(B, 1, T_img, H, W)`` — grayscale mouth ROI sequence.
            lm:  ``(B, T_lmk, 80)`` — Min-max normalized lip landmarks.

        Returns:
            ``(B, num_classes)`` logits.
        """
        if img.ndim != 5 or lm.ndim != 3:
            raise ValueError(
                f"Expected img (B,1,T_img,H,W) and lm (B,T_lmk,80); "
                f"got img{tuple(img.shape)} lm{tuple(lm.shape)}"
            )
        B, _, T_img, H, W = img.shape
        T_lmk = lm.shape[1]

        # ---- Image stream ----
        x = self.stem3d(img)                     # (B, 32, T_img, H', W')
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T_img, 32, *x.shape[3:])  # (B*T_img, 32, H', W')
        x = self.img_stage1(x)                   # (B*T_img, 64, H', W')
        x = self.img_stage2(x)                   # (B*T_img, 128, H'/2, W'/2')
        x = self.img_stage3(x)                   # (B*T_img, 256, H'/4, W'/4')
        x = self.img_pool(x).view(B, T_img, self.img_out_dim)  # (B, T_img, 256)
        img_feat = x

        # ---- Landmark stream ----
        lm_feat = self.lm_embed(lm)              # (B, T_lmk, 128)
        lm_feat = self.lm_pe(lm_feat)
        lm_feat = self.lm_transformer(lm_feat)   # (B, T_lmk, 128)

        # ---- Bidirectional cross-attention ----
        # Image attends to landmarks
        Q_img = self.proj_img_q(img_feat)        # (B, T_img, 128)
        KV_lmk = self.proj_lmk_kv(lm_feat)       # (B, T_lmk, 128)
        img_attended, _ = self.xattn_img_to_lmk(Q_img, KV_lmk, KV_lmk)
        img_fused = self.proj_img_fused(img_attended)  # (B, T_img, 256)

        # Landmarks attend to image
        Q_lmk = self.proj_lmk_q(lm_feat)         # (B, T_lmk, 128)
        KV_img = self.proj_img_kv(img_feat)      # (B, T_img, 128)
        lmk_attended, _ = self.xattn_lmk_to_img(Q_lmk, KV_img, KV_img)
        # Upsample landmark-attended to img temporal length
        if T_lmk != T_img:
            lmk_up = lmk_attended.transpose(1, 2)               # (B, 128, T_lmk)
            lmk_up = F.interpolate(lmk_up, size=T_img, mode="linear", align_corners=False)
            lmk_up = lmk_up.transpose(1, 2)                     # (B, T_img, 128)
        else:
            lmk_up = lmk_attended
        lmk_aligned = self.proj_lmk_aligned(lmk_up)             # (B, T_img, 128)

        # ---- Fuse ----
        fused = torch.cat([img_fused, lmk_aligned], dim=-1)     # (B, T_img, 384)
        fused = self.fusion_linear(fused)                       # (B, T_img, 256)

        # ---- Temporal backend ----
        seq, _ = self.bigru(fused)                              # (B, T_img, 256)
        scores = self.attn(seq).squeeze(-1)                     # (B, T_img)
        weights = F.softmax(scores, dim=1)                      # (B, T_img)
        context = (seq * weights.unsqueeze(-1)).sum(dim=1)      # (B, 256)
        return self.fc(self.dropout(context))


# ---------------------------------------------------------------------------
# Summary helper
# ---------------------------------------------------------------------------

def model_summary(
    model: MCTLR,
    img_shape: tuple[int, int, int, int, int],
    lm_shape: tuple[int, int, int],
    batch_size: int | None = None,
) -> dict[str, Any]:
    """Return trainable parameter count and estimated GFLOPs.

    Args:
        model: MCTLR instance.
        img_shape: ``(B, 1, T_img, H, W)`` for forward FLOP estimate.
        lm_shape: ``(B, T_lmk, 80)``.
        batch_size: Override batch dim (used when img_shape is a single sample).

    Returns:
        Dict with ``trainable_params``, ``approx_gflops``, ``img_shape``, ``lm_shape``.
    """
    B = batch_size if batch_size is not None else img_shape[0]
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Approximate GFLOPs — rough, based on conv and attention ops
    _, _, T_img, H, W = img_shape
    T_lmk = lm_shape[1]

    def _conv2d_flops(c_in, c_out, k, h, w, stride):
        oh = (h + 2 * (k // 2) - k) // stride + 1
        ow = (w + 2 * (k // 2) - k) // stride + 1
        return c_in * c_out * k * k * oh * ow * 2  # mul + add

    ops = 0.0
    # stem3d Conv3d: 1→32, k=(5,7,7), stride=(1,2,2)
    ops += B * 1 * 32 * 5 * 7 * 7 * T_img * (H // 2) * (W // 2)
    # stem3d MaxPool3d (negligible)
    sp_h = H // 4
    sp_w = W // 4
    in_ch = 32
    # img_stage1 (32→64, stride 1): 2 blocks × conv pairs
    for _ in range(2):
        ops += B * T_img * _conv2d_flops(in_ch, 64, 3, sp_h, sp_w, 1)  # conv1
        ops += B * T_img * _conv2d_flops(64, 64, 3, sp_h, sp_w, 1)       # conv2
        in_ch = 64
    # img_stage2 (64→128, stride 2)
    sp_h2 = sp_h // 2
    sp_w2 = sp_w // 2
    ops += B * T_img * _conv2d_flops(64, 128, 3, sp_h, sp_w, 2)  # conv1 in first block
    for _ in range(2):
        ops += B * T_img * _conv2d_flops(128, 128, 3, sp_h2, sp_w2, 1)
    # img_stage3 (128→256, stride 2)
    sp_h3 = sp_h2 // 2
    sp_w3 = sp_w2 // 2
    ops += B * T_img * _conv2d_flops(128, 256, 3, sp_h2, sp_w2, 2)
    for _ in range(2):
        ops += B * T_img * _conv2d_flops(256, 256, 3, sp_h3, sp_w3, 1)

    # Landmark stream: 2-layer transformer
    # Each self-attn layer: 4 * T_lmk * d^2 (Q,K,V,O projections) + 2 * T_lmk^2 * d
    lm_d = 128
    lm_ff = 512
    for _ in range(2):
        ops += B * (4 * T_lmk * lm_d * lm_d + 2 * T_lmk * T_lmk * lm_d)  # self-attn
        ops += B * 2 * T_lmk * lm_d * lm_ff * 2                           # FFN

    # Cross-attention (2 directions, each: 3 * T * d^2 proj + T^2 * d)
    ops += B * (3 * T_img * 128 * 128 + T_img * T_lmk * 128)  # img→lmk
    ops += B * (3 * T_lmk * 128 * 128 + T_lmk * T_img * 128)  # lmk→img

    # BiGRU: 2 layers × 2 directions × 4 gates × (input+hidden) × hidden
    gru_h = 128
    for layer in range(2):
        in_dim = 256 if layer == 0 else 256  # layer1 input is 256 (=128×2 bidirectional concat)
        for _dir in range(2):
            ops += B * T_img * 3 * (in_dim + gru_h) * gru_h * 2  # 3 gates × (mul+add)

    # Attention + FC (negligible compared to above)
    ops += B * T_img * 256 * 1 * 2   # attn
    ops += B * 256 * model.num_classes * 2  # fc

    return {
        "trainable_params": total,
        "approx_gflops": round(ops / 1e9, 4),
        "img_shape": img_shape,
        "lm_shape": lm_shape,
    }
