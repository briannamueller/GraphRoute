"""Feature extractors available to ``run_experiments.py`` configurations."""
from __future__ import annotations

from collections.abc import Callable

import torch


FeatureExtractor = Callable[[object], torch.Tensor]


# Add project-specific feature extractors here and register each one under the
# name used by graph.node_feature_source or graph.edge_feature_source.
FEATURE_EXTRACTORS: dict[str, FeatureExtractor] = {}
