"""Load tensor-backed train, validation, and test datasets."""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import TensorDataset


def load_tensor_dataset(
    data_dir: str | Path,
    split: str = "train",
) -> TensorDataset:
    """Load one split stored as a single ``(inputs, targets)`` tuple.

    Expected file: ``{data_dir}/{split}.pt``.

    Args:
        data_dir: Path to dataset directory.
        split: Dataset split name (e.g. "train", "test").

    Returns:
        TensorDataset of (features, labels).
    """
    path = Path(data_dir) / f"{split}.pt"
    stored = torch.load(path, weights_only=True)
    if not isinstance(stored, (tuple, list)) or len(stored) != 2:
        raise ValueError(
            f"{path} must contain a two-item (inputs, targets) tuple."
        )
    inputs, targets = stored
    if not torch.is_tensor(inputs) or not torch.is_tensor(targets):
        raise TypeError(f"{path} inputs and targets must both be tensors.")
    if len(inputs) != len(targets):
        raise ValueError(
            f"{path} contains {len(inputs)} inputs but {len(targets)} targets."
        )
    return TensorDataset(inputs, targets)


def load_datasets(
    data_dir: str | Path,
    dataset: str | None = None,
) -> tuple[TensorDataset, TensorDataset | None, TensorDataset]:
    """Load required train/test splits and an optional validation split.

    When ``dataset`` is supplied, the split files are loaded from
    ``{data_dir}/{dataset}``; otherwise ``data_dir`` is treated as the dataset
    folder for backwards compatibility.

    ``train.pt`` and ``test.pt`` are required. If ``validation.pt`` is absent,
    the middle return value is ``None`` so the training flow can derive a
    validation split from the training data.
    """
    data_dir = Path(data_dir)
    if dataset is not None:
        data_dir = data_dir / dataset
    train_set = load_tensor_dataset(data_dir, "train")
    validation_set = (
        load_tensor_dataset(data_dir, "validation")
        if (data_dir / "validation.pt").exists()
        else None
    )
    test_set = load_tensor_dataset(data_dir, "test")
    return train_set, validation_set, test_set
