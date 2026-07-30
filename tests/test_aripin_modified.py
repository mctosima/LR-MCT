"""Behavior checks for AripinModified model and training helpers."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import TensorDataset

from src.models.aripin_modified import AripinModified, model_summary
from src.training.train_aripin_modified import _collate_fn, augment_training_batch, train_aripin_modified


def test_model_contract() -> None:
    cases = [(10, 30, 3_854_218), (4, 40, 3_853_060)]
    for num_classes, steps, expected_params in cases:
        model = AripinModified(num_classes)
        assert model.n_params == expected_params
        summary = model_summary(model, (2, 1, steps, 80, 80))
        assert summary["trainable_params"] == expected_params
        assert summary["approx_gflops"] > 0
        logits = model(torch.zeros(2, 1, steps, 80, 80))
        assert logits.shape == (2, num_classes)


def test_model_gradients_and_validation() -> None:
    model = AripinModified(10)
    model.train()
    logits = model(torch.rand(2, 1, 3, 80, 80))
    loss = nn.CrossEntropyLoss()(logits, torch.tensor([0, 1]))
    loss.backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    for bad in (
        torch.zeros(2, 3, 3, 80, 80),
        torch.zeros(2, 1, 3, 81, 80),
        torch.zeros(2, 1, 0, 80, 80),
    ):
        try:
            model(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid input accepted: {tuple(bad.shape)}")
    try:
        model_summary(model, (2, 1, 3, 81, 80))
    except ValueError:
        pass
    else:
        raise AssertionError("invalid summary shape accepted")


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

    features = torch.rand(4, 1, 3, 80, 80)
    labels = torch.tensor([0, 1, 0, 1])
    dataset = TensorDataset(features, labels)
    with tempfile.TemporaryDirectory() as tmp:
        summary = train_aripin_modified(
            dataset,
            dataset,
            num_classes=2,
            epochs=1,
            batch_size=2,
            device="cpu",
            output_dir=Path(tmp) / "checkpoints",
            log_dir=Path(tmp) / "logs",
            run_name="test_aripin_modified",
        )
        assert summary["n_params"] == 3_852_674
        assert Path(summary["checkpoint"]).exists()
        assert Path(tmp, "logs", "test_aripin_modified.csv").exists()
        assert Path(tmp, "logs", "test_aripin_modified.json").exists()


if __name__ == "__main__":
    test_model_contract()
    test_model_gradients_and_validation()
    test_augmentation_preserves_padding()
    test_collate_and_smoke_training()
    print("AripinModified checks passed")
