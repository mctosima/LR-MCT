"""Behavior checks for AripinModifiedV2 model and training helpers."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import TensorDataset

from src.models.aripin_modified_v2 import AripinModifiedV2, model_summary
from src.training.train_aripin_modified_v2 import _collate_fn, augment_training_batch, train_aripin_modified_v2


def test_model_contract() -> None:
    cases = [(10, 30, 2_532_562), (4, 40, 2_531_788)]
    for num_classes, steps, expected_params in cases:
        model = AripinModifiedV2(num_classes)
        assert model.n_params == expected_params, f"{num_classes} classes: {model.n_params} != {expected_params}"
        summary = model_summary(model, (2, 1, steps, 80, 80))
        assert summary["trainable_params"] == expected_params
        assert summary["approx_gflops"] > 0
        x = torch.zeros(2, 1, steps, 80, 80)
        x[:, :, :2] = 0.5  # some valid frames to avoid all-black rejection
        logits = model(x)
        assert logits.shape == (2, num_classes)


def test_model_gradients_and_validation() -> None:
    model = AripinModifiedV2(10)
    model.train()
    # some non-zero frames to avoid all-black rejection
    x = torch.zeros(2, 1, 3, 80, 80)
    x[:, :, :2] = 0.5
    logits = model(x)
    loss = nn.CrossEntropyLoss()(logits, torch.tensor([0, 1]))
    loss.backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())

    # --- invalid inputs ---
    for bad in (
        torch.zeros(2, 3, 3, 80, 80),   # wrong channels
        torch.zeros(2, 1, 3, 81, 80),   # wrong spatial size
        torch.zeros(2, 1, 0, 80, 80),   # zero time
    ):
        try:
            model(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid input accepted: {tuple(bad.shape)}")

    # all-black sample rejection
    try:
        model(torch.zeros(2, 1, 3, 80, 80))
    except ValueError:
        pass
    else:
        raise AssertionError("all-black input should be rejected")

    # model_summary shape mismatch
    try:
        model_summary(model, (2, 1, 3, 81, 80))
    except ValueError:
        pass
    else:
        raise AssertionError("invalid summary shape accepted")


def test_black_tail_invariance() -> None:
    """Equivalent valid frames + different black tail lengths → same logits."""
    model = AripinModifiedV2(10)
    model.eval()
    with torch.no_grad():
        x_short = torch.zeros(2, 1, 5, 80, 80)
        x_short[:, :, :3] = 0.5
        logits_short = model(x_short)

        x_long = torch.zeros(2, 1, 8, 80, 80)
        x_long[:, :, :3] = 0.5
        logits_long = model(x_long)

        # extra frame with non-zero content → different result
        x_extra = torch.zeros(2, 1, 5, 80, 80)
        x_extra[:, :, :4] = 0.5
        logits_extra = model(x_extra)

    assert torch.allclose(logits_short, logits_long, atol=1e-5), "black tail padding should be ignored"
    assert not torch.allclose(logits_short, logits_extra, atol=1e-5), "extra valid frame must change output"


def test_augmentation_preserves_padding() -> None:
    image = torch.zeros(2, 1, 4, 80, 80)
    image[0, :, :2] = 0.5
    image[1] = 0.5
    generator = torch.Generator().manual_seed(42)
    augmented = augment_training_batch(image, generator=generator)
    assert augmented.shape == image.shape
    assert augmented.dtype == image.dtype
    assert torch.equal(augmented[0, :, 2:], torch.zeros_like(augmented[0, :, 2:]))
    assert torch.all((augmented >= 0) & (augmented <= 1))
    assert not torch.equal(augmented[0, :, :2], image[0, :, :2])


def test_collate_and_smoke_training() -> None:
    first = torch.ones(1, 2, 80, 80)
    second = torch.ones(1, 3, 80, 80)
    images, labels = _collate_fn([(first, 0), (second, 1)])
    assert images.shape == (2, 1, 3, 80, 80)
    assert labels.tolist() == [0, 1]
    assert torch.equal(images[0, :, 2:], torch.zeros_like(images[0, :, 2:]))

    # 1-epoch CPU smoke training
    features = torch.rand(4, 1, 3, 80, 80)
    labels = torch.tensor([0, 1, 0, 1])
    dataset = TensorDataset(features, labels)
    with tempfile.TemporaryDirectory() as tmp:
        summary = train_aripin_modified_v2(
            dataset,
            dataset,
            num_classes=2,
            epochs=1,
            batch_size=2,
            device="cpu",
            output_dir=Path(tmp) / "checkpoints",
            log_dir=Path(tmp) / "logs",
            run_name="test_aripin_modified_v2",
        )
        assert summary["n_params"] == 2_531_530, f"2-class param mismatch: {summary['n_params']}"
        assert Path(summary["checkpoint"]).exists()
        assert Path(tmp, "logs", "test_aripin_modified_v2.csv").exists()
        assert Path(tmp, "logs", "test_aripin_modified_v2.json").exists()
        assert summary["best_val_acc"] >= 0
        assert summary["final_val_loss"] >= 0


if __name__ == "__main__":
    test_model_contract()
    test_model_gradients_and_validation()
    test_black_tail_invariance()
    test_augmentation_preserves_padding()
    test_collate_and_smoke_training()
    print("AripinModifiedV2 checks passed")
