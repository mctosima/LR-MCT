"""Plain assertions for MCT-LR2 model, dataset, and augmentation.

Run with:  .venv/bin/python tests/test_mct_lr2.py
"""

import sys
from pathlib import Path

# Ensure repo root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch import Tensor, nn

from src.data.mct_dataset import (
    MCTFusionDataset,
    Sample,
    make_loso_folds,
    split_train_val,
)
from src.models.mct_lr2 import MCTLR2, model_summary
from src.training.train_mct_lr2 import augment_training_batch

# ── parameter counts ──────────────────────────────────────────────────

m10 = MCTLR2(10)
m4 = MCTLR2(4)
assert m10.n_params == 7_564_468, f"K=10: expected 7_564_468, got {m10.n_params}"
assert m4.n_params == 7_562_926, f"K=4:  expected 7_562_926, got {m4.n_params}"
print(f"Param counts: K=10={m10.n_params:,}, K=4={m4.n_params:,}")

# ── eval forward ──────────────────────────────────────────────────────

for m, k in ((m10, 10), (m4, 4)):
    m.eval()
    B, T_i, T_l = 2, 3, 4
    img = torch.randn(B, 1, T_i, 80, 80)
    lm = torch.randn(B, T_l, 80)
    out = m(img, lm)
    assert tuple(out.shape) == (B, k), f"Expected ({B},{k}), got {tuple(out.shape)}"
    cl, sp = m(img, lm, return_aux=True, grl_scale=0.5)
    assert tuple(cl.shape) == (B, k)
    assert tuple(sp.shape) == (B, 8)
    print(f"K={k}: eval forward OK (logits {tuple(out.shape)}, aux speaker {tuple(sp.shape)})")

# ── training forward + backward ───────────────────────────────────────

m10.train()
img = torch.randn(2, 1, 3, 80, 80)
lm = torch.randn(2, 4, 80)
cl, sp = m10(img, lm, return_aux=True, grl_scale=0.3)
class_loss = nn.CrossEntropyLoss()(cl, torch.randint(0, 10, (2,)))
sp_loss = nn.CrossEntropyLoss()(sp, torch.randint(0, 8, (2,)))
loss = class_loss + 0.1 * sp_loss
loss.backward()

grad_keys = set()
for name, param in m10.named_parameters():
    if param.grad is not None and param.grad.abs().sum() > 0:
        prefix = name.split(".")[0]
        if prefix in ("img_stage1", "img_stage2", "img_stage3"):
            grad_keys.add("img_blocks")
        else:
            grad_keys.add(prefix)
# Verify key components receive gradients
required = {"img_stem", "img_blocks", "img_lstm", "img_proj",
            "lm_ln", "lm_gru", "lm_proj",
            "cross_img2lm", "cross_lm2img",
            "pool_img", "pool_lm",
            "classifier", "speaker_head"}
missing = required - grad_keys
assert not missing, f"No gradients in: {missing}"
print("Backward: all required components receive finite gradients")

# ── eval mode disables modality dropout ───────────────────────────────

m10.eval()
img = torch.randn(1, 1, 3, 80, 80)
lm = torch.randn(1, 4, 80)
# Run multiple times — should be deterministic in eval mode with fixed dropout
base = m10(img, lm)
for _ in range(5):
    assert torch.equal(base, m10(img, lm)), "Eval forward not deterministic (modality dropout active?)"
print("Eval mode: deterministic (modality dropout disabled)")

# ── Augmentation preserves zero padding ───────────────────────────────

img = torch.zeros(2, 1, 5, 80, 80)
img[0, :, :3] = 0.5          # sample 0: 3 real frames, 2 zeros
img[1, :, :5] = 0.3          # sample 1: all real
lm = torch.rand(2, 6, 80)
img_aug, _ = augment_training_batch(img.clone(), lm)
# Check zero-padding frames stayed zero
for i in range(2):
    frames_all_zero = img[i, 0].abs().sum(dim=(1, 2)) == 0
    if frames_all_zero.any():
        for t in frames_all_zero.nonzero(as_tuple=True)[0]:
            assert (img_aug[i, 0, t] == 0).all(), f"Sample {i} frame {t} was zero-padded but got nonzero after aug"
print("Augmentation: zero-padding frames restored correctly")

# ── input validation ──────────────────────────────────────────────────

def _should_fail(fn, *args, **kw):
    try:
        fn(*args, **kw)
        return False  # unexpected pass
    except (ValueError, AssertionError):
        return True   # correctly raised

m = MCTLR2(10)
# Bad shapes
assert _should_fail(m, torch.randn(2, 1, 3, 80, 80), torch.randn(2, 4, 81)),  "bad lm dim"
assert _should_fail(m, torch.randn(2, 3, 3, 80, 80), torch.randn(2, 4, 80)), "bad img channels"
assert _should_fail(m, torch.randn(2, 1, 3, 80, 80), torch.randn(3, 4, 80)), "batch mismatch"
assert _should_fail(m, torch.randn(2, 1, 0, 80, 80), torch.randn(2, 4, 80)), "zero img time"
assert _should_fail(m, torch.randn(2, 1, 3, 80, 80), torch.randn(2, 0, 80)),  "zero lm time"
# Bad params
assert _should_fail(MCTLR2, 0),           "num_classes=0"
assert _should_fail(MCTLR2, 10, 1),        "num_speakers=1"
assert _should_fail(MCTLR2, 10, 8, (80, 80), 0.6),  "modality dropout out of range"
# model_summary bad shapes
assert _should_fail(model_summary, m, (1, 3, 30, 80, 80), (1, 40, 80)),  "bad img channels in summary"
assert _should_fail(model_summary, m, (1, 1, 30, 80, 80), (1, 40, 81)),  "bad lm dim in summary"
print("Input validation: all bad inputs rejected correctly")

# ── LOSO folds ────────────────────────────────────────────────────────

speakers = [f"speaker_{i}" for i in range(8)]
samples: list[Sample] = []
for sp in speakers:
    for cls_name in ("01 Words", "11 Phrase"):
        samples.append(("roi.npy", "lm.npy", cls_name, sp, f"{sp}__vid"))

folds = make_loso_folds(samples)
assert len(folds) == 8, f"Expected 8 folds, got {len(folds)}"
all_test = set()
for i, (test_sp, val_sp, tr, va, te) in enumerate(folds):
    tr_sp = {s[3] for s in tr}
    assert len(tr_sp) == 6, f"Fold {i}: expected 6 train speakers, got {len(tr_sp)}"
    assert {s[3] for s in va} == {val_sp}, f"Fold {i}: val speaker mismatch"
    assert {s[3] for s in te} == {test_sp}, f"Fold {i}: test speaker mismatch"
    assert tr_sp.isdisjoint({val_sp, test_sp}), f"Fold {i}: train shares val/test"
    all_test.add(test_sp)
assert len(all_test) == 8, f"Expected 8 unique test speakers, got {len(all_test)}"
print(f"LOSO: 8 folds validated ({len(all_test)} unique test speakers)")

# ── split_train_val backward compat ────────────────────────────────────

ci = [c for _, _, c, _, _ in samples]
si = [sp for _, _, _, sp, _ in samples]
ds1, ds2 = split_train_val(samples, ci, si, seed=42)
assert not ds1.include_speaker and not ds2.include_speaker
assert len(ds1.labels) > 0 and len(ds2.labels) > 0

ds3, ds4 = split_train_val(samples, ci, si, seed=42, include_speaker=True)
assert ds3.include_speaker and ds4.include_speaker
assert ds3.class_to_idx == ds4.class_to_idx
assert ds3.speaker_to_idx == ds4.speaker_to_idx
assert len(ds3.speaker_to_idx) == 8
assert len(ds3.speaker_labels) > 0
print("split_train_val: backward compat and speaker mode verified")

print("\n--- All tests passed ---")
