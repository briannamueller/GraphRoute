"""Configuration-driven experiment and result-management behavior."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from graphroute.config import GraphRouteConfig
from graphroute.experiment import (
    build_model_pool,
    expand_experiments,
    experiment_id,
    is_completed,
    load_experiments,
    result_path,
    save_failure,
    save_result,
)


def _specification():
    return {
        "dataset": "demo",
        "num_classes": 2,
        "base": {"models": ["linear"]},
        "sweep": {"seed": [0, 1], "graph.k": [3, 7]},
    }


def test_sweep_expands_a_deterministic_cartesian_product():
    configurations = expand_experiments(_specification())

    assert [(cfg.graph.k, cfg.seed) for cfg in configurations] == [
        (3, 0), (3, 1), (7, 0), (7, 1),
    ]
    assert all(cfg.base.models == ["linear"] for cfg in configurations)


def test_yaml_loads_the_same_validated_sweep(tmp_path):
    path = tmp_path / "experiment.yaml"
    path.write_text(
        "dataset: demo\n"
        "num_classes: 2\n"
        "base:\n"
        "  models: [linear]\n"
        "sweep:\n"
        "  graph.k: [2, 4]\n"
    )

    assert [cfg.graph.k for cfg in load_experiments(path)] == [2, 4]


def test_shipped_configuration_defines_one_experiment():
    path = Path(__file__).parents[1] / "configs" / "experiment.yaml"

    configurations = load_experiments(path)

    assert len(configurations) == 1
    assert configurations[0].seed == 0
    assert configurations[0].graph.k == 5
    assert configurations[0].gnn.arch == "gat"


@pytest.mark.parametrize("sweep,match", [
    ({"unknown": [1]}, "Unknown sweep parameter"),
    ({"graph.k": []}, "nonempty list"),
    ({"graph.k": 5}, "nonempty list"),
])
def test_invalid_sweep_definitions_are_rejected(sweep, match):
    specification = _specification()
    specification["sweep"] = sweep

    with pytest.raises((TypeError, ValueError), match=match):
        expand_experiments(specification)


def test_registry_builds_the_named_ordered_pool():
    cfg = GraphRouteConfig(
        dataset="demo", num_classes=2,
        base={"models": ["wide", "narrow", "wide"]})
    registry = {
        "narrow": lambda sample, run: nn.Linear(sample.numel(), run.num_classes),
        "wide": lambda sample, run: nn.Sequential(
            nn.Linear(sample.numel(), 8), nn.Linear(8, run.num_classes)),
    }

    models = build_model_pool(cfg, torch.ones(4), registry)

    assert [len(model) if isinstance(model, nn.Sequential) else 0
            for model in models] == [2, 0, 2]
    assert models[0] is not models[2]


def test_registry_reports_unknown_names():
    cfg = GraphRouteConfig(dataset="demo", base={"models": ["missing"]})

    with pytest.raises(ValueError, match="registered names: linear"):
        build_model_pool(
            cfg, torch.ones(4),
            {"linear": lambda sample, run: nn.Linear(sample.numel(), 2)})


def test_experiment_identity_repeats_with_the_same_seed_and_models():
    cfg = GraphRouteConfig(
        dataset="demo", num_classes=2,
        base={"models": ["linear"]})

    torch.manual_seed(cfg.seed)
    first = [nn.Linear(4, 2)]
    torch.manual_seed(cfg.seed)
    second = [nn.Linear(4, 2)]

    assert experiment_id(cfg, first) == experiment_id(cfg, second)
    changed = cfg.model_copy(update={"seed": 1})
    assert experiment_id(cfg, first) != experiment_id(changed, first)


def test_completed_and_failed_results_have_resume_semantics(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "graphroute.experiment._graphroute_version", lambda: "test-version")
    cfg = GraphRouteConfig(dataset="demo")
    destination = result_path(tmp_path, cfg, "abc123")
    assert destination == tmp_path / "demo" / "abc123.json"
    assert not is_completed(destination)

    save_failure(
        destination, run_id="abc123", cfg=cfg,
        error=RuntimeError("training failed"))
    assert not is_completed(destination)
    failed = json.loads(destination.read_text())
    assert failed["status"] == "failed"
    assert failed["graphroute_version"] == "test-version"
    assert failed["error"] == {
        "type": "RuntimeError", "message": "training failed"}

    save_result(
        destination, run_id="abc123", cfg=cfg,
        metrics={"accuracy": np.float64(0.75), "count": torch.tensor(4)})
    assert is_completed(destination)
    completed = json.loads(destination.read_text())
    assert completed["status"] == "completed"
    assert completed["graphroute_version"] == "test-version"
    assert completed["configuration"]["dataset"] == "demo"
    assert completed["metrics"] == {"accuracy": 0.75, "count": 4}
