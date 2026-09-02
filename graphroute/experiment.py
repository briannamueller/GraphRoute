"""Configuration-driven experiment and sweep utilities."""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import os
import tempfile
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from pydantic import ValidationError

from graphroute.config import BaseConfig, GNNConfig, GraphConfig, GraphRouteConfig
from graphroute.pool_cache import automatic_model_ids, safe_component

ModelFactory = Callable[[Any, GraphRouteConfig], nn.Module]
ModelRegistry = Mapping[str, ModelFactory]


def _graphroute_version() -> str:
    try:
        return version("graphroute")
    except PackageNotFoundError:
        return "unknown"


def _known_sweep_path(path: str) -> bool:
    parts = path.split(".")
    if len(parts) == 1:
        return parts[0] in GraphRouteConfig.model_fields and parts[0] not in {
            "base", "graph", "gnn"
        }
    if len(parts) != 2:
        return False
    group, field = parts
    groups = {"base": BaseConfig, "graph": GraphConfig, "gnn": GNNConfig}
    return group in groups and field in groups[group].model_fields


def _set_sweep_value(values: dict, path: str, value: Any) -> None:
    if not _known_sweep_path(path):
        raise ValueError(
            f"Unknown sweep parameter {path!r}; use a GraphRouteConfig field or "
            "a dotted nested field such as 'graph.k'."
        )
    parts = path.split(".")
    if len(parts) == 1:
        values[parts[0]] = value
        return
    group, field = parts
    nested = values.setdefault(group, {})
    if not isinstance(nested, dict):
        raise ValueError(f"Configuration group {group!r} must be a mapping.")
    nested[field] = value


def expand_experiments(specification: Mapping[str, Any]) -> list[GraphRouteConfig]:
    """Expand a base configuration and dotted-path grid into validated runs."""
    if not isinstance(specification, Mapping):
        raise TypeError("The experiment configuration must be a mapping.")
    base = copy.deepcopy(dict(specification))
    sweep = base.pop("sweep", {})
    if sweep is None:
        sweep = {}
    if not isinstance(sweep, Mapping):
        raise TypeError("The 'sweep' entry must be a mapping of fields to lists.")

    paths = sorted(sweep)
    choices = []
    for path in paths:
        values = sweep[path]
        if not isinstance(values, list) or not values:
            raise ValueError(
                f"Sweep parameter {path!r} must contain a nonempty list of values."
            )
        if not _known_sweep_path(path):
            _set_sweep_value({}, path, values[0])
        choices.append(values)

    combinations = itertools.product(*choices) if choices else [()]
    resolved = []
    for combination in combinations:
        values = copy.deepcopy(base)
        for path, value in zip(paths, combination):
            _set_sweep_value(values, path, value)
        try:
            resolved.append(GraphRouteConfig.model_validate(values))
        except ValidationError as error:
            selected = dict(zip(paths, combination))
            raise ValueError(
                f"Invalid experiment configuration for sweep values {selected}: "
                f"{error}"
            ) from error
    return resolved


def load_experiments(path: str | Path) -> list[GraphRouteConfig]:
    """Read a YAML experiment file and return every validated configuration."""
    path = Path(path)
    try:
        specification = yaml.safe_load(path.read_text())
    except yaml.YAMLError as error:
        raise ValueError(f"Could not parse YAML configuration {path}: {error}") from error
    if specification is None:
        raise ValueError(f"Experiment configuration {path} is empty.")
    return expand_experiments(specification)


def build_model_pool(
    cfg: GraphRouteConfig,
    sample_input: Any,
    registry: ModelRegistry,
) -> list[nn.Module]:
    """Construct the ordered pool named by ``cfg.base.models``."""
    names = cfg.base.models
    if not names:
        raise ValueError("Set base.models to a nonempty list of registered model names.")
    unknown = [name for name in names if name not in registry]
    if unknown:
        available = ", ".join(sorted(registry)) or "none"
        raise ValueError(
            f"Unknown model names {unknown}; registered names: {available}."
        )
    models = [registry[name](sample_input, cfg) for name in names]
    if not all(isinstance(model, nn.Module) for model in models):
        raise TypeError("Every model registry factory must return a torch.nn.Module.")
    return models


def experiment_id(cfg: GraphRouteConfig, models: list[nn.Module]) -> str:
    """Identify a resolved configuration and its initialized model pool."""
    payload = {
        "configuration": cfg.model_dump(mode="json"),
        "models": automatic_model_ids(models),
        "graphroute": _graphroute_version(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def result_path(
    results_dir: str | Path,
    cfg: GraphRouteConfig,
    run_id: str,
) -> Path:
    """Return the per-dataset JSON path for one resolved experiment."""
    return Path(results_dir) / safe_component(cfg.dataset) / f"{run_id}.json"


def is_completed(path: str | Path) -> bool:
    """Whether ``path`` contains a completed run result."""
    path = Path(path)
    if not path.exists():
        return False
    try:
        return json.loads(path.read_text()).get("status") == "completed"
    except (OSError, json.JSONDecodeError, AttributeError):
        return False


def _json_default(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.item() if value.numel() == 1 else value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot store {type(value).__name__} in a result file.")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp", delete=False)
    temporary = Path(handle.name)
    try:
        with handle:
            json.dump(payload, handle, indent=2, sort_keys=True,
                      default=_json_default)
            handle.write("\n")
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def save_result(
    path: str | Path,
    *,
    run_id: str,
    cfg: GraphRouteConfig,
    metrics: Mapping[str, Any],
) -> None:
    """Atomically store one completed configuration and its metrics."""
    _write_json(Path(path), {
        "run_id": run_id,
        "status": "completed",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "graphroute_version": _graphroute_version(),
        "configuration": cfg.model_dump(mode="json"),
        "metrics": dict(metrics),
    })


def save_failure(
    path: str | Path,
    *,
    run_id: str,
    cfg: GraphRouteConfig,
    error: BaseException,
) -> None:
    """Atomically record a failed run without marking it complete."""
    _write_json(Path(path), {
        "run_id": run_id,
        "status": "failed",
        "failed_at": datetime.now(timezone.utc).isoformat(),
        "graphroute_version": _graphroute_version(),
        "configuration": cfg.model_dump(mode="json"),
        "error": {"type": type(error).__name__, "message": str(error)},
    })


__all__ = [
    "ModelFactory",
    "ModelRegistry",
    "build_model_pool",
    "expand_experiments",
    "experiment_id",
    "is_completed",
    "load_experiments",
    "result_path",
    "save_failure",
    "save_result",
]
