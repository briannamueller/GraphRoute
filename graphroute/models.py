"""Small built-in base models and name-to-factory helpers.

A name is ``<arch><width>`` -- ``mlp64``, ``cnn32`` -- so a heterogeneous pool is
just a list of names. To use your own models, pass them to
:func:`graphroute.run.fit_graphroute` or add their factories to a model registry.
"""
from __future__ import annotations

import re
from typing import Callable

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, width: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU(),
        )
        self.head = nn.Linear(width, num_classes)

    def forward(self, x):
        return self.head(self.net(x.flatten(1)))


class SmallCNN(nn.Module):
    """Two conv blocks then a linear head. Expects [N, C, H, W]."""

    def __init__(self, in_shape: tuple[int, ...], width: int, num_classes: int):
        super().__init__()
        c = in_shape[0]
        self.features = nn.Sequential(
            nn.Conv2d(c, width, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(width, width * 2, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(width * 2, num_classes)

    def forward(self, x):
        return self.head(self.features(x).flatten(1))


def build_factory(name: str, in_shape: tuple[int, ...], num_classes: int) -> Callable[[], nn.Module]:
    """Resolve one ``<arch><width>`` name to a callable returning a fresh model."""
    m = re.fullmatch(r"([a-z]+)(\d+)", name.strip().lower())
    if not m:
        raise ValueError(f"Unrecognised model name {name!r}; expected e.g. 'mlp64' or 'cnn32'.")
    arch, width = m.group(1), int(m.group(2))
    if arch == "mlp":
        in_dim = int(torch.tensor(in_shape).prod())
        return lambda: MLP(in_dim, width, num_classes)
    if arch == "cnn":
        if len(in_shape) != 3:
            raise ValueError(f"'cnn' needs [C, H, W] input, got shape {tuple(in_shape)}.")
        return lambda: SmallCNN(in_shape, width, num_classes)
    raise ValueError(f"Unknown architecture {arch!r}; known: mlp, cnn.")


def build_pool_factories(names: list[str], in_shape, num_classes: int):
    return [build_factory(n, tuple(in_shape), num_classes) for n in names]
