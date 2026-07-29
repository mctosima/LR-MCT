"""MCT-LR2 --- residual image encoder, motion-augmented landmark encoder,
cross-modal attention fusion, and speaker-adversarial regularisation.

Architecture
------------
Image stream:
  3-stage residual CNN (1→32→64→128→256) → BiLSTM-2(192) → Linear+LN → 256-d per frame

Landmark stream:
  coordinate + inter-frame delta → LayerNorm → BiGRU-3(192) → Linear+LN → 256-d per frame

Fusion:
  two pre-norm cross-attention blocks (image↔landmark) → additive attention pool
  → concat → head → classifier

Speaker-invariance:
  gradient reversal on fused representation → speaker classifier

Reference: plan at local://mct-lr2-model-plan.md
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


# ── gradient reversal ──────────────────────────────────────────────────

class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, scale: float) -> Tensor:
        ctx.scale = scale
        return x

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, None]:
        return (-ctx.scale * grad_output, None)


def _grl(x: Tensor, scale: float) -> Tensor:
    return _GradientReversal.apply(x, scale)


# ── building blocks ────────────────────────────────────────────────────

class _ResBlock(nn.Module):
    """Two 3×3 convs with BatchNorm + SiLU and optional stride-2 shortcut."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.shortcut: nn.Module
        if in_ch != out_ch or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        residual = self.shortcut(x)
        out = F.silu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.silu(out + residual)


class _CrossBlock(nn.Module):
    """Pre-norm cross-attention with residual FFN.

    Performs  q_out = FFN(LN(cross_attn(LN(q), LN(kv)) + q)) + q
    """

    def __init__(self, dim: int = 256, heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm_ff = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, query: Tensor, key_value: Tensor) -> Tensor:
        q_norm = self.norm_q(query)
        kv_norm = self.norm_kv(key_value)
        attn_out, _ = self.attn(q_norm, kv_norm, kv_norm, need_weights=False)
        out = query + attn_out
        return out + self.ff(self.norm_ff(out))


class _AdditiveAttention(nn.Module):
    """Tanh-based additive attention pool over time dimension."""

    def __init__(self, dim: int = 256) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim // 2)
        self.scorer = nn.Linear(dim // 2, 1)

    def forward(self, x: Tensor) -> Tensor:
        scores = self.scorer(torch.tanh(self.proj(x))).squeeze(-1)  # (B, T)
        weights = F.softmax(scores, dim=1).unsqueeze(-1)            # (B, T, 1)
        return (x * weights).sum(dim=1)                              # (B, dim)


# ── MCT-LR2 ────────────────────────────────────────────────────────────

class MCTLR2(nn.Module):
    """Multimodal visual speech recognition with cross-modal attention.

    Args:
        num_classes: Number of output classes (10 words or 4 phrases).
        num_speakers: Speaker vocabulary size for adversarial branch.
        img_size: Grayscale ROI spatial size (must be divisible by 16).
        modality_dropout: Dropout probability for each modality stream
            during training. Must satisfy ``0 <= modality_dropout < 0.5``.
    """

    def __init__(
        self,
        num_classes: int,
        num_speakers: int = 8,
        img_size: tuple[int, int] = (80, 80),
        modality_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if num_speakers < 2:
            raise ValueError("num_speakers must be at least 2")
        if not (0 <= modality_dropout < 0.5):
            raise ValueError("modality_dropout must be in [0, 0.5)")

        self.num_classes = num_classes
        self.num_speakers = num_speakers
        self.img_size = img_size
        self.modality_dropout = modality_dropout

        # ---- image spatial stream (per-frame) --------------------------
        self.img_stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.MaxPool2d(2),
        )
        self.img_stage1 = nn.Sequential(
            _ResBlock(32, 64, stride=2),
            _ResBlock(64, 64),
        )
        self.img_stage2 = nn.Sequential(
            _ResBlock(64, 128, stride=2),
            _ResBlock(128, 128),
        )
        self.img_stage3 = nn.Sequential(
            _ResBlock(128, 256, stride=2),
            _ResBlock(256, 256),
        )
        self.img_pool = nn.AdaptiveAvgPool2d(1)  # → (B*T, 256, 1, 1)

        # ---- image temporal stream -------------------------------------
        self.img_lstm = nn.LSTM(
            256, 192, num_layers=2, dropout=0.3,
            batch_first=True, bidirectional=True,
        )
        self.img_proj = nn.Sequential(
            nn.Linear(384, 256),
            nn.LayerNorm(256),
        )

        # ---- landmark motion stream ------------------------------------
        self.lm_ln = nn.LayerNorm(160)
        self.lm_gru = nn.GRU(
            160, 192, num_layers=3, dropout=0.3,
            batch_first=True, bidirectional=True,
        )
        self.lm_proj = nn.Sequential(
            nn.Linear(384, 256),
            nn.LayerNorm(256),
        )

        # ---- fusion ----------------------------------------------------
        self.cross_img2lm = _CrossBlock()
        self.cross_lm2img = _CrossBlock()

        # ---- pooling & classification ----------------------------------
        self.pool_img = _AdditiveAttention(256)
        self.pool_lm = _AdditiveAttention(256)
        self.head = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Dropout(0.4),
        )
        self.classifier = nn.Linear(256, num_classes)

        # ---- speaker-invariance branch ---------------------------------
        self.speaker_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_speakers),
        )

    # -------------------------------------------------------------------
    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # -------------------------------------------------------------------
    def forward(
        self,
        img: Tensor,
        lm: Tensor,
        *,
        return_aux: bool = False,
        grl_scale: float = 1.0,
    ) -> Tensor | tuple[Tensor, Tensor]:
        if img.ndim != 5 or img.shape[1] != 1 or tuple(img.shape[-2:]) != self.img_size:
            raise ValueError(
                f"Expected img shape (B, 1, T, {self.img_size[0]}, {self.img_size[1]}), got {tuple(img.shape)}"
            )
        if lm.ndim != 3 or lm.shape[-1] != 80:
            raise ValueError(f"Expected lm shape (B, T, 80), got {tuple(lm.shape)}")
        if img.shape[0] != lm.shape[0]:
            raise ValueError(f"Batch sizes differ: img={img.shape[0]}, lm={lm.shape[0]}")
        if img.shape[2] < 1:
            raise ValueError("Image time dimension is zero")
        if lm.shape[1] < 1:
            raise ValueError("Landmark time dimension is zero")
        if not (0 <= grl_scale <= 1):
            raise ValueError("grl_scale must be in [0, 1]")

        B, _, T_img, H, W = img.shape
        T_lm = lm.shape[1]

        # ---- image spatial stream (per-frame) --------------------------
        frames = img.permute(0, 2, 1, 3, 4).reshape(B * T_img, 1, H, W)
        frames = self.img_stem(frames)       # (B*T, 32, H/2, W/2)
        frames = self.img_stage1(frames)     # (B*T, 64, H/4, W/4)
        frames = self.img_stage2(frames)     # (B*T, 128, H/8, W/8)
        frames = self.img_stage3(frames)     # (B*T, 256, H/16, W/16)
        frames = self.img_pool(frames)       # (B*T, 256, 1, 1)
        img_seq = frames.squeeze(-1).squeeze(-1).reshape(B, T_img, 256)  # (B, T_img, 256)

        # ---- image temporal stream -------------------------------------
        img_lstm_out, _ = self.img_lstm(img_seq)     # (B, T_img, 384)
        img_feat = self.img_proj(img_lstm_out)        # (B, T_img, 256)

        # ---- landmark motion stream ------------------------------------
        delta = torch.zeros_like(lm)
        delta[:, 1:] = lm[:, 1:] - lm[:, :-1]
        lm_cat = torch.cat([lm, delta], dim=-1)        # (B, T_lm, 160)
        lm_cat = self.lm_ln(lm_cat)
        lm_gru_out, _ = self.lm_gru(lm_cat)            # (B, T_lm, 384)
        lm_feat = self.lm_proj(lm_gru_out)             # (B, T_lm, 256)

        # ---- modality dropout (training only) --------------------------
        if self.training and self.modality_dropout > 0 and B > 0:
            sample = torch.rand(B, device=img.device)
            # 0 = keep both, 1 = drop img, 2 = drop lm
            choice = (sample > 1 - self.modality_dropout).long() + \
                     (sample > 1 - 2 * self.modality_dropout).long()
            for i in range(B):
                if choice[i] == 1:
                    img_feat[i] = 0
                elif choice[i] == 2:
                    lm_feat[i] = 0

        # ---- cross-modal fusion ----------------------------------------
        fused_img = self.cross_img2lm(img_feat, lm_feat)    # (B, T_img, 256)
        fused_lm = self.cross_lm2img(lm_feat, img_feat)     # (B, T_lm,  256)

        # ---- pooling ---------------------------------------------------
        pooled_img = self.pool_img(fused_img)               # (B, 256)
        pooled_lm = self.pool_lm(fused_lm)                  # (B, 256)
        fused = torch.cat([pooled_img, pooled_lm], dim=-1)  # (B, 512)
        fused = self.head(fused)                            # (B, 256)

        # ---- classifier ------------------------------------------------
        class_logits = self.classifier(fused)                # (B, num_classes)

        if return_aux:
            rev = _grl(fused, grl_scale)
            speaker_logits = self.speaker_head(rev)
            return class_logits, speaker_logits

        return class_logits


# ── model summary ──────────────────────────────────────────────────────

def model_summary(
    model: MCTLR2,
    img_shape: tuple[int, int, int, int, int],
    lm_shape: tuple[int, int, int],
) -> dict[str, Any]:
    """Return trainable parameter count and approximate batch GFLOPs."""
    if len(img_shape) != 5 or img_shape[1] != 1 or tuple(img_shape[-2:]) != model.img_size:
        raise ValueError(f"img_shape must be (B, 1, T, {model.img_size[0]}, {model.img_size[1]})")
    if len(lm_shape) != 3 or lm_shape[2] != 80:
        raise ValueError("lm_shape must be (B, T, 80)")

    B, _, T_img, H, W = img_shape
    T_lm = lm_shape[1]
    ops = 0.0

    def _conv_flops(ci: int, co: int, k: int, hi: int, wi: int) -> float:
        return 2 * ci * co * k * k * hi * wi

    def _linear_flops(ci: int, co: int) -> float:
        return 2 * ci * co

    def _rnn_flops(inp: int, hid: int, layers: int, dirs: int, t: int, gates: int) -> float:
        total = 0.0
        for layer in range(layers):
            i = inp if layer == 0 else hid * dirs
            total += 2.0 * B * t * dirs * gates * hid * (i + hid + 1)
        return total

    # --- image spatial ---
    H, W = model.img_size
    # stem
    ops += _conv_flops(1, 32, 3, H, W)               # conv
    ops += 2 * 32 * H * W                              # BN?  plan omits norm ops, but let's be consistent.
    H, W = H // 2, W // 2

    # stage 1: 32→64 stride 2, 64→64 stride 1
    ops += _conv_flops(32, 64, 3, H, W) + _conv_flops(64, 64, 3, H // 2, W // 2)  # block 0
    ops += _conv_flops(32, 64, 1, H, W)                # block 0 shortcut
    H, W = H // 2, W // 2
    ops += _conv_flops(64, 64, 3, H, W) + _conv_flops(64, 64, 3, H, W)  # block 1

    # stage 2: 64→128 stride 2, 128→128 stride 1
    ops += _conv_flops(64, 128, 3, H, W) + _conv_flops(128, 128, 3, H // 2, W // 2)
    ops += _conv_flops(64, 128, 1, H, W)
    H, W = H // 2, W // 2
    ops += _conv_flops(128, 128, 3, H, W) + _conv_flops(128, 128, 3, H, W)

    # stage 3: 128→256 stride 2, 256→256 stride 1
    ops += _conv_flops(128, 256, 3, H, W) + _conv_flops(256, 256, 3, H // 2, W // 2)
    ops += _conv_flops(128, 256, 1, H, W)
    H, W = H // 2, W // 2
    ops += _conv_flops(256, 256, 3, H, W) + _conv_flops(256, 256, 3, H, W)

    ops *= B * T_img  # per-frame → whole batch

    # --- image temporal ---
    ops += _rnn_flops(256, 192, 2, 2, T_img, 4)
    ops += B * T_img * _linear_flops(384, 256)          # img_proj Linear
    # LayerNorm omitted

    # --- landmark temporal ---
    ops += _rnn_flops(160, 192, 3, 2, T_lm, 3)
    ops += B * T_lm * _linear_flops(384, 256)          # lm_proj Linear

    # --- cross-attention (2 blocks) ---
    for _ in range(2):
        # MHA: 4 projections of 256x256
        ops += 4 * B * _linear_flops(256, 256) * max(T_img, T_lm)
        # attention score: Q*K^T  (query × key^T)
        ops += 2 * (B * T_img * T_lm * 256 + B * T_lm * T_img * 256)
        # FFN: 256→512, 512→256
        ops += B * T_img * _linear_flops(256, 512)
        ops += B * T_lm * _linear_flops(512, 256)

    # --- additive attention pooling (×2) ---
    ops += 2 * B * T_img * _linear_flops(256, 128)
    ops += 2 * B * T_img * _linear_flops(128, 1)
    ops += 2 * B * T_lm * _linear_flops(256, 128)
    ops += 2 * B * T_lm * _linear_flops(128, 1)

    # --- head ---
    ops += B * _linear_flops(512, 256)
    ops += B * _linear_flops(256, model.num_classes)

    # --- speaker head (for GFLOP estimate, count it) ---
    ops += B * _linear_flops(256, 128)
    ops += B * _linear_flops(128, model.num_speakers)

    return {
        "trainable_params": model.n_params,
        "approx_gflops": round(ops / 1e9, 4),
        "img_shape": img_shape,
        "lm_shape": lm_shape,
    }
