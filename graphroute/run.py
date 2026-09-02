"""Fit and evaluate GraphRoute."""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from graphroute.config import GraphRouteConfig
from graphroute.graph import build_graph
from graphroute.gnn import build_gnn
from graphroute.losses import compute_meta_labels, compute_regression_meta_labels
from graphroute.training import FallbackModel, compute_pair_features, train_gnn
from graphroute.pool_cache import (PoolArtifact, automatic_model_ids, cached_pool,
                                   fingerprint, fingerprint_pool, in_memory_pool,
                                   pool_directory)
from graphroute.pool import (
    apply_calibrators,
    calibrate_pool,
    collect_pool_embeddings,
    split_train_meta,
    train_pool,
    train_pool_oof,
)


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _labels(loader: DataLoader) -> torch.Tensor:
    return torch.cat([y for _, y in loader])


def _inputs(loader: DataLoader) -> torch.Tensor:
    return torch.cat([x for x, _ in loader]).flatten(1).float()


def build_features(
    source: str,
    pool_outputs: torch.Tensor,
    raw: Optional[torch.Tensor],
    embeddings: Optional[list[torch.Tensor]],
) -> torch.Tensor:
    """Build node or edge features from predictions, inputs, or embeddings.

    Classification predictions arrive as ``[N, M, C]`` probabilities and are
    flattened. Scalar-regression predictions already have decision-space shape
    ``[N, M]``.
    """
    if pool_outputs.ndim == 3:
        decision = pool_outputs.flatten(1)
    elif pool_outputs.ndim == 2:
        decision = pool_outputs
    else:
        raise ValueError(
            "Pool outputs must be [N, M, C] for classification or [N, M] for "
            f"scalar regression; got {tuple(pool_outputs.shape)}.")
    if source == "decision_space":
        return decision
    if source == "feature_space":
        if raw is None:
            raise ValueError("feature_space needs the raw inputs.")
        return raw
    if source == "hybrid":
        if raw is None:
            raise ValueError("hybrid needs the raw inputs.")
        return torch.cat([decision, raw], dim=1)
    if source in ("embedding_concat", "embedding_mean"):
        if not embeddings:
            raise ValueError(f"{source} needs pool embeddings.")
        if source == "embedding_concat":
            return torch.cat(embeddings, dim=1)
        widths = {e.shape[1] for e in embeddings}
        if len(widths) > 1:
            raise ValueError(
                f"embedding_mean averages across pool models, so their penultimate "
                f"widths must match; this pool has {sorted(widths)}. Use "
                f"embedding_concat for a heterogeneous pool.")
        return torch.stack(embeddings, dim=0).mean(dim=0)
    raise ValueError(f"Unknown feature source {source!r}.")


def _pool_outputs(cfg, pool, loader, device, calibrators, want_raw, want_emb,
                  logits):
    """Pool predictions and any additional requested representations.

    ``logits`` overrides the forward pass -- that is how out-of-fold predictions
    reach the meta split, since the final models were retrained on those rows and
    predicting them again would be in-sample.
    """
    if cfg.task == "regression":
        predictions = logits.float()
    else:
        predictions = (apply_calibrators(calibrators, logits)
                       if calibrators is not None
                       else torch.softmax(logits, dim=-1))
    raw = _inputs(loader) if want_raw else None
    emb = (collect_pool_embeddings(pool.load_models(device), loader, device)
           if want_emb else None)
    return predictions, raw, emb


def _template_factories(models: list[torch.nn.Module]):
    """Clone caller-owned templates whenever isolated model state is needed."""
    return [lambda template=model: copy.deepcopy(template) for model in models]


def _output_name(split: str) -> str:
    """Name outputs by their role within the explicitly named dataset."""
    return f"{split}_logits"


def _dataset_outputs(pool: PoolArtifact, split: str, dataset: Dataset,
                     loader: DataLoader, device: torch.device, *, task: str,
                     cache_outputs: bool = True) -> torch.Tensor:
    """Collect outputs under the caller's explicit dataset identity."""
    target = pool if cache_outputs else pool.for_data(output_directory=None)
    return target.cached_outputs(_output_name(split), loader, device, task=task)


@dataclass
class GraphRouteModel:
    """The fitted graph ensemble and the training data needed for inference."""

    cfg: GraphRouteConfig
    pool: PoolArtifact
    gnn: torch.nn.Module
    history: dict
    train_node_features: torch.Tensor
    train_edge_features: torch.Tensor
    train_decision_space: torch.Tensor
    train_labels: torch.Tensor
    train_meta_labels: torch.Tensor
    calibrators: list | None
    competence_scale: torch.Tensor | None
    device: torch.device
    collate_fn: Callable | None = None

    def _dataset_parts(self, dataset: Dataset, split: str, *, cache_outputs: bool):
        loader = DataLoader(dataset, batch_size=256, shuffle=False,
                            collate_fn=self.collate_fn)
        labels = _labels(loader)
        g = self.cfg.graph
        sources = {g.node_feature_source, g.edge_feature_source}
        want_raw = bool(sources & {"feature_space", "hybrid"})
        want_emb = bool(sources & {"embedding_mean", "embedding_concat"})
        logits = _dataset_outputs(
            self.pool, split, dataset, loader, self.device,
            task=self.cfg.task,
            cache_outputs=cache_outputs)
        values = _pool_outputs(self.cfg, self.pool, loader, self.device,
                               self.calibrators, want_raw, want_emb, logits)
        node = build_features(g.node_feature_source, *values)
        edge = build_features(g.edge_feature_source, *values)
        ds = values[0].reshape(values[0].shape[0], -1)
        return labels, node, edge, ds

    @torch.no_grad()
    def predict(self, dataset: Dataset, *, split: str = "test",
                cache_outputs: bool = True) -> dict:
        """Predict a dataset with mutually independent evaluation queries."""
        from graphroute.selection import (compute_selection_matrix,
                                          evaluate_ensemble,
                                          evaluate_ensemble_regression)

        labels, node, edge, ds = self._dataset_parts(
            dataset, split, cache_outputs=cache_outputs)
        g = self.cfg.graph
        data, gmeta = build_graph(
            self.train_node_features, self.train_labels,
            self.train_decision_space, self.train_meta_labels,
            eval_features=node, eval_labels=labels, eval_ds=ds,
            eval_edge_features=edge, eval_meta=None, eval_type="test",
            k=g.k, neighbor_mode=g.neighbor_mode, weight_mode=g.weight_mode,
            num_classes=self.cfg.num_classes,
            train_edge_features=self.train_edge_features, task=self.cfg.task)
        data = data.to(self.device)
        eval_mask = data["sample"].test_mask.bool()
        args = self.cfg.gnn_namespace()
        combined_ds = gmeta["combined_ds"].to(self.device)
        if args.gnn_output_head == "concat_mlp":
            pair = compute_pair_features(
                combined_ds, self.cfg.num_classes,
                pair_confidence=args.gnn_output_head_pair_confidence,
                pair_competence=args.gnn_output_head_pair_competence,
                data=data, meta=gmeta["combined_meta"].to(self.device))
            if pair is not None:
                data["sample"].pair_feats = pair.to(self.device)
        self.gnn.eval()
        scores = self.gnn(data)[eval_mask]
        if self.cfg.gnn.fallback != "uniform":
            fallback = FallbackModel(self.cfg.gnn.fallback,
                                     self.train_meta_labels, self.train_labels)
            missing = torch.relu(scores).sum(dim=1) == 0
            if missing.any():
                replacement = fallback(data, eval_mask)
                scores[missing] = replacement[missing].to(self.device)

        eval_ds = combined_ds[eval_mask]
        selection, fallback_rows = compute_selection_matrix(
            scores, args.gnn_ens_combination_mode,
            args.gnn_voting_weight_space)
        common = {
            "selection_scores": scores.cpu(),
            "effective_pool_size": (selection > 0).float().sum(dim=1).cpu(),
            "fallback": fallback_rows.cpu(),
        }
        if self.cfg.task == "regression":
            predictions = evaluate_ensemble_regression(
                scores, eval_ds.float(), args.gnn_voting_weight_space)
            return {**common, "predictions": predictions.cpu()}
        hard = eval_ds.view(-1, len(self.pool), self.cfg.num_classes).argmax(2)
        probabilities, predictions = evaluate_ensemble(
            scores, eval_ds, self.cfg.num_classes,
            args.gnn_ens_combination_mode, args.gnn_voting_weight_space,
            hard_preds=hard)
        return {**common, "probabilities": probabilities.cpu(),
                "predictions": predictions.cpu()}

    def evaluate(self, dataset: Dataset, *, split: str = "test") -> dict:
        """Return unprefixed metrics for a validation or test dataset."""
        predicted = self.predict(dataset, split=split)
        labels = _labels(DataLoader(dataset, batch_size=256, shuffle=False,
                                    collate_fn=self.collate_fn))
        metrics = {
            "effective_pool_size": predicted["effective_pool_size"].mean().item(),
            "fallback_rate": predicted["fallback"].float().mean().item(),
        }
        if self.cfg.task == "regression":
            mse = torch.nn.functional.mse_loss(
                predicted["predictions"].float(), labels.float()).item()
            return {"mse": mse, "rmse": mse ** 0.5, **metrics}
        labels = labels.long()
        hard = predicted["predictions"]
        accuracy = (hard == labels).float().mean().item()
        recalls = [(hard[labels == c] == c).float().mean().item()
                   for c in range(self.cfg.num_classes) if (labels == c).any()]
        return {"accuracy": accuracy,
                "balanced_accuracy": float(np.mean(recalls)) if recalls else 0.0,
                **metrics}


def _pool_training_kwargs(cfg: GraphRouteConfig) -> dict:
    b = cfg.base
    return dict(num_classes=cfg.num_classes, batch_size=b.batch_size,
                max_epochs=b.epochs, patience=b.es_patience, lr=b.lr,
                optimizer_name=b.optimizer, weight_decay=b.weight_decay,
                weighted_by_class=b.weighted_by_class, es_metric=b.es_metric,
                task=cfg.task)


def _default_code_identity() -> dict:
    package_dir = Path(__file__).resolve().parent
    training_code = {
        name: (package_dir / name).read_text()
        for name in ("pool.py", "models.py")
    }
    try:
        from importlib.metadata import version
        package_version = version("graphroute")
    except Exception:
        package_version = "unknown"
    return {"graphroute": package_version,
            "base_training_code": fingerprint(training_code)}


def fit_graphroute(
    cfg: GraphRouteConfig,
    train_set: Dataset,
    validation_set: Optional[Dataset] = None,
    models: Optional[list[torch.nn.Module]] = None,
    *,
    cache_dir: Optional[str | Path] = None,
    pool: Optional[PoolArtifact] = None,
    collate_fn: Callable | None = None,
) -> GraphRouteModel:
    """Fit GraphRoute, training or reusing a pool when one is not supplied.

    Args:
        cfg: GraphRoute configuration.
        train_set: An indexable PyTorch dataset returning ``(inputs, target)``.
        validation_set: Optional validation dataset. When omitted, GraphRoute
            derives one from ``train_set`` using ``cfg.val_ratio``.
        models: Caller-owned model templates. GraphRoute leaves these unchanged
            and deep-copies them for every isolated OOF/final training job.
        cache_dir: Directory for persistent trained-pool and output reuse.
            Defaults to ``pool_cache``; pass an empty string to disable persistence.
        pool: Advanced prebuilt pool artifact carrying training outputs.
        collate_fn: Optional PyTorch batching function for structured samples. It
            must return ``(batch_inputs, batch_targets)``; ``batch_inputs`` must
            support ``.to(device)`` and be accepted by every model as one object.
    """
    persistent_pool = cache_dir != ""
    resolved_cache_dir = Path("pool_cache") if cache_dir is None else Path(cache_dir)
    derived_val_ratio = cfg.val_ratio if validation_set is None else None
    sources = {cfg.graph.node_feature_source, cfg.graph.edge_feature_source}

    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    if validation_set is None:
        train_set, validation_set = split_train_meta(
            train_set, val_ratio=cfg.val_ratio, seed=cfg.seed, task=cfg.task)

    if pool is None:
        if models is None:
            from graphroute.models import build_pool_factories
            if not cfg.base.models:
                raise ValueError(
                    "Pass models or name architectures in base.models.")
            sample_x, _ = train_set[0]
            built = build_pool_factories(
                cfg.base.models, sample_x.shape, cfg.num_classes)
            models = [factory() for factory in built]
        if not models:
            raise ValueError("models must contain at least one nn.Module.")
        if not all(isinstance(model, torch.nn.Module) for model in models):
            raise TypeError("Every entry in models must be a torch.nn.Module.")
        model_factories = _template_factories(models)
        try:
            model_ids = automatic_model_ids(models)
        except ValueError:
            if persistent_pool:
                raise
            model_ids = [
                f"{type(model).__module__}.{type(model).__qualname__}:{index}"
                for index, model in enumerate(models)
            ]

        pool_fp = fingerprint_pool(
            model_ids=model_ids,
            base_config={"base": cfg.base.model_dump(), "task": cfg.task,
                         "num_classes": cfg.num_classes,
                         "derived_val_ratio": derived_val_ratio},
            seed=cfg.seed, code_identity=_default_code_identity())
        pool_kw = _pool_training_kwargs(cfg)
        if cfg.base.split_mode == "oof_stacking":
            meta_set = train_set

            def train():
                trained, oof, _ = train_pool_oof(
                    model_factories, train_set, validation_set, device,
                    n_folds=cfg.base.oof_folds, seed=cfg.seed,
                    collate_fn=collate_fn, **pool_kw)
                return trained, oof
        else:
            base_set, meta_set = split_train_meta(
                train_set, val_ratio=0.5, seed=cfg.seed, task=cfg.task)

            def train():
                return (train_pool(model_factories, base_set, validation_set,
                                   device, collate_fn=collate_fn, **pool_kw), None)

        if not persistent_pool:
            trained_models, oof = train()
            pool = in_memory_pool(trained_models, model_ids=model_ids,
                                  fingerprint_value=pool_fp)
            pool.oof_logits = oof
        else:
            directory = pool_directory(
                resolved_cache_dir, cfg.dataset, pool_fp)
            pool = cached_pool(
                directory, model_factories, train,
                fingerprint_value=pool_fp, model_ids=model_ids,
                require_oof=cfg.base.split_mode == "oof_stacking",
                data_id=cfg.dataset)
    else:
        meta_set = train_set
        if pool.training_outputs is None:
            raise ValueError(
                "A supplied PoolArtifact must carry the training outputs for these rows.")

    print(f"[GraphRoute] {cfg.dataset}: {len(pool)} pool models, "
          f"device={device}, seed={cfg.seed}")
    meta_loader = DataLoader(meta_set, batch_size=256, shuffle=False,
                             collate_fn=collate_fn)
    val_loader = DataLoader(validation_set, batch_size=256, shuffle=False,
                            collate_fn=collate_fn)
    labels = {"train": _labels(meta_loader), "validation": _labels(val_loader)}
    training_logits = pool.training_outputs
    if training_logits is None:
        training_logits = (pool.load_oof()
                           if cfg.base.split_mode == "oof_stacking"
                           else _dataset_outputs(
                               pool, "train", meta_set, meta_loader, device,
                               task=cfg.task))
    if training_logits is None:
        raise RuntimeError("OOF stacking produced no training outputs.")
    if training_logits.shape[0] != len(meta_set):
        raise RuntimeError(
            f"Cached training outputs contain {training_logits.shape[0]} rows, "
            f"but dataset {cfg.dataset!r} currently contains {len(meta_set)}. "
            "Use a new dataset name or remove the stale cache.")
    validation_logits = _dataset_outputs(
        pool, "validation", validation_set, val_loader, device,
        task=cfg.task)

    calibrators = None
    if cfg.graph.pool_calibrate:
        _, calibrators = calibrate_pool(
            training_logits, labels["train"], cfg.graph.calib_method)

    g = cfg.graph
    want_raw = bool(sources & {"feature_space", "hybrid"})
    want_emb = bool(sources & {"embedding_mean", "embedding_concat"})
    values = {
        "train": _pool_outputs(cfg, pool, meta_loader, device, calibrators,
                               want_raw, want_emb, training_logits),
        "validation": _pool_outputs(cfg, pool, val_loader, device, calibrators,
                                    want_raw, want_emb, validation_logits),
    }
    node = {k: build_features(g.node_feature_source, *v) for k, v in values.items()}
    edge = {k: build_features(g.edge_feature_source, *v) for k, v in values.items()}
    ds = {k: v[0].reshape(v[0].shape[0], -1) for k, v in values.items()}

    scale = None
    if cfg.task == "regression":
        needs = cfg.loss_target == "meta_labels" or cfg.gnn.pair_competence == "gain"
        if needs:
            scale = labels["train"].float().std(correction=0)
            meta_train = compute_regression_meta_labels(
                ds["train"], labels["train"], scale=scale)
            meta_val = compute_regression_meta_labels(
                ds["validation"], labels["validation"], scale=scale)
        else:
            meta_train = torch.zeros_like(ds["train"])
            meta_val = torch.zeros_like(ds["validation"])
    else:
        meta_train = compute_meta_labels(
            values["train"][0].argmax(-1), labels["train"])
        meta_val = compute_meta_labels(
            values["validation"][0].argmax(-1), labels["validation"])

    data, gmeta = build_graph(
        node["train"], labels["train"], ds["train"], meta_train,
        eval_features=node["validation"], eval_labels=labels["validation"],
        eval_ds=ds["validation"], eval_edge_features=edge["validation"],
        eval_meta=meta_val, eval_type="val", k=g.k,
        neighbor_mode=g.neighbor_mode, weight_mode=g.weight_mode,
        num_classes=cfg.num_classes, train_edge_features=edge["train"], task=cfg.task)

    # Pool training consumes random numbers on a miss but not a hit. Resetting at
    # the GNN boundary makes the fitted ensemble independent of cache state.
    seed_everything(cfg.seed + 1)
    cfg.validate_head()
    gnn = build_gnn(
        cfg.gnn.arch,
        **cfg.gnn_kwargs(input_dim=node["train"].shape[1], out_dim=len(pool),
                         pair_feat_dim=cfg.pair_feat_dim()))
    gnn, history = train_gnn(gnn, data, gmeta, cfg.gnn_namespace(), device)
    return GraphRouteModel(
        cfg=cfg, pool=pool, gnn=gnn, history=history,
        train_node_features=node["train"], train_edge_features=edge["train"],
        train_decision_space=ds["train"], train_labels=labels["train"],
        train_meta_labels=meta_train, calibrators=calibrators,
        competence_scale=scale, device=device, collate_fn=collate_fn)
