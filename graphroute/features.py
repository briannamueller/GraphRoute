"""Feature extraction helpers for sample representations."""

from __future__ import annotations

import torch


BUILTIN_FEATURE_SOURCES = frozenset({
    "decision_space",
    "feature_space",
    "embedding_mean",
    "embedding_concat",
    "hybrid",
})


def extract_feature_slice(
    inputs,
    *,
    input_index: int,
    start: int,
    stop: int,
) -> torch.Tensor:
    """Select a contiguous feature group from one model-input field."""
    if isinstance(inputs, torch.Tensor):
        if input_index != 0:
            raise IndexError("A single-input batch has only input_index 0.")
        field = inputs
    else:
        field = inputs[input_index]
    return field[..., start:stop]
