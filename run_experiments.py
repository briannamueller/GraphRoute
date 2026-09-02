"""Run one GraphRoute configuration or an optional Cartesian sweep.

Model factories are defined in ``model_registry.py``. Fixed settings and the
optional sweep are defined in ``configs/experiment.yaml``. Run:

    python run_experiments.py --config configs/experiment.yaml

Completed configurations are skipped unless ``--force`` is supplied.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from graphroute.data import load_datasets
from graphroute.experiment import (
    build_model_pool,
    experiment_id,
    is_completed,
    load_experiments,
    result_path,
    save_failure,
    save_result,
)
from graphroute.run import fit_graphroute, seed_everything
from model_registry import MODEL_REGISTRY


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a GraphRoute experiment or sweep.")
    parser.add_argument(
        "--config", required=True, type=Path,
        help="YAML file containing the configuration and optional sweep.")
    parser.add_argument(
        "--results-dir", type=Path, default=Path("results"),
        help="Parent directory for per-run JSON results.")
    parser.add_argument(
        "--force", action="store_true",
        help="Rerun configurations that already completed.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    configurations = load_experiments(args.config)

    datasets = {}
    failures = 0
    for index, cfg in enumerate(configurations):
        data_key = (cfg.data_dir, cfg.dataset)
        if data_key not in datasets:
            datasets[data_key] = load_datasets(*data_key)
        train_set, validation_set, test_set = datasets[data_key]

        # The initialized model pool is part of both the cache and run identity.
        seed_everything(cfg.seed)
        models = build_model_pool(cfg, train_set[0][0], MODEL_REGISTRY)
        run_id = experiment_id(cfg, models)
        destination = result_path(args.results_dir, cfg, run_id)
        label = f"[{index + 1}/{len(configurations)}] {run_id}"

        if not args.force and is_completed(destination):
            print(f"{label}: already completed; skipping")
            continue

        print(f"{label}: running")
        try:
            trained = fit_graphroute(
                cfg,
                train_set,
                validation_set=validation_set,
                models=models,
            )
            metrics = trained.evaluate(test_set)
            save_result(
                destination, run_id=run_id, cfg=cfg, metrics=metrics)
            print(f"{label}: saved {destination}")
            for name, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"  {name:24} {value:.4f}")
        except Exception as error:
            save_failure(
                destination, run_id=run_id, cfg=cfg, error=error)
            print(f"{label}: failed: {error}", file=sys.stderr)
            failures += 1

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
