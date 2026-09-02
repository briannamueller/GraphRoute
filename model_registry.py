"""Model factories available to ``run_experiments.py`` configurations."""
from __future__ import annotations

from graphroute.experiment import ModelRegistry
from graphroute.models import MLP


def _mlp(width: int):
    def build(sample_input, cfg):
        return MLP(int(sample_input.numel()), width, cfg.num_classes)
    return build


# Add project-specific factories here. Each factory receives one unbatched input
# sample and the resolved configuration and returns a new PyTorch model.
MODEL_REGISTRY: ModelRegistry = {
    "mlp64": _mlp(64),
    "mlp128": _mlp(128),
    "mlp256": _mlp(256),
}
