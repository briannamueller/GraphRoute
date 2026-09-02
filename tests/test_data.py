"""Tensor split files used by the standalone experiment workflow."""

import pytest
import torch

from graphroute.data import load_datasets, load_tensor_dataset


def _save_split(directory, name, n=4):
    inputs = torch.arange(n * 3, dtype=torch.float32).reshape(n, 3)
    targets = torch.arange(n, dtype=torch.long)
    torch.save((inputs, targets), directory / f"{name}.pt")
    return inputs, targets


def test_one_file_contains_inputs_and_targets(tmp_path):
    inputs, targets = _save_split(tmp_path, "train")

    dataset = load_tensor_dataset(tmp_path, "train")

    assert torch.equal(dataset.tensors[0], inputs)
    assert torch.equal(dataset.tensors[1], targets)


def test_validation_is_none_when_its_file_is_absent(tmp_path):
    _save_split(tmp_path, "train")
    _save_split(tmp_path, "test")

    train_set, validation_set, test_set = load_datasets(tmp_path)

    assert len(train_set) == 4
    assert validation_set is None
    assert len(test_set) == 4


def test_named_dataset_is_loaded_below_the_data_root(tmp_path):
    dataset_dir = tmp_path / "my_dataset"
    dataset_dir.mkdir()
    _save_split(dataset_dir, "train")
    _save_split(dataset_dir, "test", n=2)

    train_set, validation_set, test_set = load_datasets(
        tmp_path, "my_dataset")

    assert len(train_set) == 4
    assert validation_set is None
    assert len(test_set) == 2


def test_explicit_validation_file_is_loaded(tmp_path):
    _save_split(tmp_path, "train")
    _save_split(tmp_path, "validation", n=2)
    _save_split(tmp_path, "test")

    _, validation_set, _ = load_datasets(tmp_path)

    assert validation_set is not None
    assert len(validation_set) == 2


def test_train_and_test_files_are_required(tmp_path):
    _save_split(tmp_path, "train")

    with pytest.raises(FileNotFoundError, match="test.pt"):
        load_datasets(tmp_path)


def test_split_file_must_contain_matching_tensor_pair(tmp_path):
    torch.save({"inputs": torch.ones(2, 3)}, tmp_path / "train.pt")
    with pytest.raises(ValueError, match="two-item"):
        load_tensor_dataset(tmp_path, "train")

    torch.save((torch.ones(2, 3), torch.ones(3)), tmp_path / "train.pt")
    with pytest.raises(ValueError, match="2 inputs but 3 targets"):
        load_tensor_dataset(tmp_path, "train")
