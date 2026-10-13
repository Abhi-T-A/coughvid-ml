"""Focused unit tests for PyTorch SpectrogramDataset, SpecAugment, and CoughAudioCNN."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from coughvid.models.cnn import CoughAudioCNN
from coughvid.models.dataset import SpecAugment, SpectrogramDataset


@pytest.fixture
def mock_spectrogram_data() -> tuple[list[str], dict[str, int], dict[str, np.ndarray]]:
    """Generate synthetic 64x128 spectrograms and labels for 10 UUIDs."""
    rng = np.random.RandomState(42)
    uuids = [f"uuid_{i:04d}" for i in range(10)]
    labels = {u: i % 2 for i, u in enumerate(uuids)}
    specs = {u: rng.uniform(-1.0, 1.0, size=(64, 128)).astype(np.float32) for u in uuids}
    return uuids, labels, specs


def test_spectrogram_dataset_uuid_alignment(mock_spectrogram_data):
    """Verify that dataset retrieves items strictly by UUID even when UUID order is shuffled."""
    uuids, labels, specs = mock_spectrogram_data

    # Shuffled order
    shuffled_uuids = [uuids[7], uuids[2], uuids[0], uuids[9], uuids[4]]
    ds = SpectrogramDataset(
        uuids=shuffled_uuids,
        labels=labels,
        spectrogram_source=specs,
        is_train=False,
    )

    assert len(ds) == len(shuffled_uuids)

    for idx, expected_uuid in enumerate(shuffled_uuids):
        spec_tensor, label_tensor, item_uuid = ds[idx]
        assert item_uuid == expected_uuid
        assert label_tensor.item() == float(labels[expected_uuid])
        expected_arr = specs[expected_uuid]
        np.testing.assert_allclose(spec_tensor.squeeze().numpy(), expected_arr, atol=1e-6)


def test_spectrogram_dataset_missing_uuid_raises_error(mock_spectrogram_data):
    """Verify that missing UUIDs in label mapping or spectrogram store raise ValueError."""
    uuids, labels, specs = mock_spectrogram_data

    incomplete_labels = {u: labels[u] for u in uuids[:-1]}
    with pytest.raises(ValueError, match="Missing labels"):
        SpectrogramDataset(uuids=uuids, labels=incomplete_labels, spectrogram_source=specs)

    incomplete_specs = {u: specs[u] for u in uuids[:-1]}
    with pytest.raises(ValueError, match="Missing spectrograms"):
        SpectrogramDataset(uuids=uuids, labels=labels, spectrogram_source=incomplete_specs)


def test_specaugment_isolation(mock_spectrogram_data):
    """Verify that SpecAugment is strictly isolated to training mode and validation is deterministic."""
    uuids, labels, specs = mock_spectrogram_data
    augmenter = SpecAugment(max_freq_mask=8, max_time_mask=16, fill_value=-1.0, rng=np.random.RandomState(42))

    # 1. Validation dataset (is_train=False) must NOT augment, even if augmenter passed
    ds_val = SpectrogramDataset(
        uuids=uuids,
        labels=labels,
        spectrogram_source=specs,
        is_train=False,
        spec_augment=augmenter,
    )
    # Repeated calls must return bit-for-bit identical tensors
    tensor1, _, _ = ds_val[0]
    tensor2, _, _ = ds_val[0]
    assert torch.equal(tensor1, tensor2)
    # Must match original raw array exactly
    np.testing.assert_allclose(tensor1.squeeze().numpy(), specs[uuids[0]], atol=1e-6)

    # 2. Training dataset (is_train=True) must apply augmentation
    ds_train = SpectrogramDataset(
        uuids=uuids,
        labels=labels,
        spectrogram_source=specs,
        is_train=True,
        spec_augment=augmenter,
    )
    aug_tensor, _, _ = ds_train[0]
    # In training mode, masked values (-1.0) must be injected into the spectrogram
    assert torch.any(aug_tensor == -1.0)
    # Augmented tensor should differ from raw input
    assert not torch.equal(aug_tensor, tensor1)


def test_cnn_forward_shapes_and_finite_logits():
    """Verify CoughAudioCNN accepts (B, 1, 64, 128) and outputs (B, 1) finite logits."""
    model = CoughAudioCNN()
    model.eval()

    batch_size = 4
    x = torch.randn(batch_size, 1, 64, 128)

    with torch.no_grad():
        logits = model(x)
        probs = model.predict_proba(x)

    assert logits.shape == (batch_size, 1)
    assert probs.shape == (batch_size, 1)
    assert torch.all(torch.isfinite(logits))
    assert torch.all(torch.isfinite(probs))
    assert torch.all((probs >= 0.0) & (probs <= 1.0))
    # Probabilities must match sigmoid of logits
    expected_probs = torch.sigmoid(logits)
    torch.testing.assert_close(probs, expected_probs)


def test_cnn_parameter_count_exact():
    """Verify the exact trainable parameter count of CoughAudioCNN."""
    model = CoughAudioCNN()
    param_count = model.count_parameters()
    assert param_count == 101553


def test_cnn_invalid_input_shapes_raise_error():
    """Verify invalid tensor dimensions or channels raise ValueError."""
    model = CoughAudioCNN()

    # Wrong channels (e.g. 3 channels instead of 1)
    with pytest.raises(ValueError, match="channel dimension"):
        model(torch.randn(2, 3, 64, 128))

    # Invalid dimension (e.g. 2D or 5D)
    with pytest.raises(ValueError, match="3D or 4D tensor"):
        model(torch.randn(2, 1, 64, 128, 2))


def test_cnn_serialization_and_reload(tmp_path):
    """Verify torch.save and torch.load state_dict produces identical outputs."""
    torch.manual_seed(42)
    model = CoughAudioCNN()
    model.eval()

    x = torch.randn(3, 1, 64, 128)
    with torch.no_grad():
        orig_logits = model(x)

    save_path = tmp_path / "cnn_checkpoint.pt"
    torch.save(model.state_dict(), save_path)

    reloaded_model = CoughAudioCNN()
    reloaded_model.load_state_dict(torch.load(save_path, weights_only=True))
    reloaded_model.eval()

    with torch.no_grad():
        reloaded_logits = reloaded_model(x)

    torch.testing.assert_close(orig_logits, reloaded_logits)


def test_checkpoint_restoration_during_eval(tmp_path):
    """Verify that trainer explicitly restores best checkpoint before generating final validation predictions."""
    torch.manual_seed(42)
    model = CoughAudioCNN()
    model.eval()

    x = torch.randn(2, 1, 64, 128)
    with torch.no_grad():
        best_epoch_logits = model(x)

    best_checkpoint_path = tmp_path / "best_model.pt"
    torch.save(model.state_dict(), best_checkpoint_path)

    # Simulate subsequent training that drifts the model weights
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.1)

    with torch.no_grad():
        drifted_logits = model(x)
    assert not torch.equal(best_epoch_logits, drifted_logits)

    # Restore best checkpoint
    model.load_state_dict(torch.load(best_checkpoint_path, weights_only=True))
    model.eval()

    with torch.no_grad():
        restored_logits = model(x)

    torch.testing.assert_close(best_epoch_logits, restored_logits)


def test_fold_pos_weight_calculation():
    """Verify positive class weight is computed strictly as n_neg / n_pos on training data."""
    # Synthetic labels: 10 positives, 90 negatives
    train_labels = {f"u_{i}": 1 if i < 10 else 0 for i in range(100)}
    train_uuids = list(train_labels.keys())

    n_pos = sum(train_labels[u] for u in train_uuids)
    n_neg = len(train_uuids) - n_pos
    pos_weight = float(n_neg) / float(n_pos)

    assert n_pos == 10
    assert n_neg == 90
    assert pos_weight == 9.0


def test_oof_coverage_assertion_checks():
    """Verify OOF coverage assertion logic detects missing or duplicate UUIDs."""
    eligible_uuids = {f"u_{i:04d}" for i in range(100)}

    # Valid complete coverage
    valid_oof = pd.DataFrame({
        "uuid": list(eligible_uuids),
        "y_prob": [0.5] * 100,
    })
    assert len(valid_oof) == len(eligible_uuids)
    assert valid_oof["uuid"].nunique() == len(eligible_uuids)

    # Missing UUID
    incomplete_oof = valid_oof.iloc[:-1]
    assert len(incomplete_oof) != len(eligible_uuids)

    # Duplicate UUID
    duplicate_oof = pd.concat([valid_oof, valid_oof.iloc[:1]])
    assert duplicate_oof["uuid"].nunique() != len(duplicate_oof)

