"""Base-model training and prediction collection."""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import (
    KFold,
    ShuffleSplit,
    StratifiedKFold,
    StratifiedShuffleSplit,
)
from torch.utils.data import DataLoader, Dataset, Subset

from graphroute.calibration import get_calibrator


# ── Loss construction ───────────────────────────────────────────────────

class _BalancedBCEWithLogits(nn.Module):
    """Binary cross-entropy with logits using a positive-class weight.

    For a two-output model this reads column 1 only; column 0 is untrained and
    argmax still selects correctly.
    """

    def __init__(self, pos_weight: float):
        super().__init__()
        self.pos_weight = torch.tensor([pos_weight], dtype=torch.float32)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.dim() > 1 and logits.size(1) == 2:
            logits = logits[:, 1]
        targets = targets.float()
        return F.binary_cross_entropy_with_logits(
            logits, targets,
            pos_weight=self.pos_weight.to(logits.device),
        )


def build_loss_fn(
    num_classes: int,
    weighted: bool,
    device: torch.device,
    train_dataset: Dataset | None = None,
) -> nn.Module:
    """Build classification loss.

    Binary (num_classes=2): BCEWithLogits with optional pos_weight.
    Multi-class: CrossEntropy with optional inverse-frequency weights.
    """
    if num_classes == 2:
        if not weighted or train_dataset is None:
            return _BalancedBCEWithLogits(pos_weight=1.0)
        counts = torch.zeros(num_classes, dtype=torch.float)
        for _, y in train_dataset:
            lbl = int(y.item()) if torch.is_tensor(y) else int(y)
            if 0 <= lbl < counts.numel():
                counts[lbl] += 1.0
        pos_weight = (counts[0] / counts[1].clamp(min=1.0)).clamp(max=50.0)
        return _BalancedBCEWithLogits(pos_weight=pos_weight.to(device))

    if not weighted or train_dataset is None:
        return nn.CrossEntropyLoss()

    counts = torch.zeros(num_classes, dtype=torch.float)
    for _, y in train_dataset:
        lbl = int(y.item()) if torch.is_tensor(y) else int(y)
        if 0 <= lbl < counts.numel():
            counts[lbl] += 1.0
    if counts.sum() == 0:
        return nn.CrossEntropyLoss()
    safe = counts.clamp(min=1.0)
    weights = (safe.sum() / safe.numel()) / safe
    return nn.CrossEntropyLoss(weight=weights.to(device))


def build_regression_loss_fn() -> nn.Module:
    """Build regression loss (MSE)."""
    return nn.MSELoss()


# ── Epoch helpers ───────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    task: str = "classification",
) -> dict:
    """Single training epoch."""
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        if task == "regression":
            loss = loss_fn(logits.squeeze(-1), y.float())
        else:
            loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            total_loss += loss.item() * y.size(0)
            total += y.size(0)
            if task == "classification":
                preds = logits.argmax(dim=1)
                total_correct += (preds == y).sum().item()

    avg_loss = total_loss / max(total, 1)
    stats = {"loss": avg_loss}
    if task == "classification":
        stats["acc"] = total_correct / max(total, 1)
    return stats


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
    task: str = "classification",
) -> dict:
    """Evaluate loss (and accuracy for classification)."""
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    per_class_correct: dict[int, int] = {}
    per_class_total: dict[int, int] = {}

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        if task == "regression":
            loss = loss_fn(logits.squeeze(-1), y.float())
        else:
            loss = loss_fn(logits, y)

        total_loss += loss.item() * y.size(0)
        total += y.size(0)
        if task == "classification":
            preds = logits.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            for cls in y.unique():
                m = y == cls
                c = int(cls)
                per_class_correct[c] = per_class_correct.get(c, 0) + int((preds[m] == cls).sum())
                per_class_total[c] = per_class_total.get(c, 0) + int(m.sum())

    avg_loss = total_loss / max(total, 1)
    stats = {"loss": avg_loss}
    if task == "classification":
        stats["acc"] = total_correct / max(total, 1)
        # Balanced accuracy: mean per-class recall, so a classifier that ignores
        # a rare class cannot score well by getting the common one right.
        recalls = [per_class_correct.get(c, 0) / n for c, n in per_class_total.items() if n]
        stats["bacc"] = sum(recalls) / len(recalls) if recalls else 0.0
    return stats


# ── Single-model training ──────────────────────────────────────────────

def fit_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    *,
    max_epochs: int = 300,
    patience: int = 20,
    lr: float = 0.0005,
    optimizer_name: str = "Adam",
    weight_decay: float = 5e-4,
    task: str = "classification",
    num_classes: int = 10,
    weighted_by_class: bool = True,
    es_metric: str = "val_loss",
) -> tuple[int, float, nn.Module]:
    """Train one base model with early stopping.

    Args:
        model: The nn.Module to train (will be moved to device).
        train_loader: Training data loader.
        val_loader: Validation data loader.
        device: Compute device.
        max_epochs: Maximum training epochs.
        patience: Early stopping patience.
        lr: Learning rate.
        optimizer_name: "Adam" or "SGD".
        weight_decay: L2 regularization.
        task: "classification" or "regression".
        num_classes: Number of classes (classification only).
        weighted_by_class: Use class-weighted loss (classification only).
        es_metric: Early stopping metric ("val_loss", "val_acc", "val_bacc").

    Returns:
        (best_epoch, best_metric_value, trained_model)
    """
    if task == "regression" and es_metric != "val_loss":
        raise ValueError(
            "Regression base models support early stopping on val_loss.")
    model = model.to(device)

    if task == "regression":
        loss_fn = build_regression_loss_fn()
        eval_loss_fn = build_regression_loss_fn()
    else:
        loss_fn = build_loss_fn(num_classes, weighted_by_class, device,
                                train_dataset=train_loader.dataset)
        eval_loss_fn = build_loss_fn(num_classes, False, device)

    if optimizer_name.upper() == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                    momentum=0.9, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                     weight_decay=weight_decay)

    best_state = None
    best_metric = -float("inf")
    best_epoch, stale_epochs = 0, 0

    for epoch in range(1, max_epochs + 1):
        train_one_epoch(model, train_loader, device, optimizer, loss_fn, task=task)
        val_stats = evaluate(model, val_loader, device, eval_loss_fn, task=task)

        if es_metric == "val_acc" and "acc" in val_stats:
            metric_val = val_stats["acc"]
        elif es_metric == "val_bacc" and "bacc" in val_stats:
            metric_val = val_stats["bacc"]
        else:  # val_loss
            metric_val = -val_stats["loss"]

        if metric_val > best_metric:
            best_metric = metric_val
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1

        if stale_epochs >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return best_epoch, best_metric, model


# ── Pool training ───────────────────────────────────────────────────────

def train_pool(
    model_factories: list[Callable[[], nn.Module]],
    train_dataset: Dataset,
    val_dataset: Dataset,
    device: torch.device,
    *,
    batch_size: int = 64,
    max_epochs: int = 300,
    patience: int = 20,
    lr: float = 0.0005,
    optimizer_name: str = "Adam",
    weight_decay: float = 5e-4,
    task: str = "classification",
    num_classes: int = 10,
    weighted_by_class: bool = True,
    es_metric: str = "val_loss",
    collate_fn: Callable | None = None,
) -> list[nn.Module]:
    """Train the full pool of base models.

    Args:
        model_factories: List of callables, each returning a fresh nn.Module.
        train_dataset: Training dataset.
        val_dataset: Validation dataset (for early stopping).
        device: Compute device.
        batch_size: Batch size for training.
        max_epochs: Maximum epochs per model.
        patience: Early stopping patience.
        lr: Learning rate.
        optimizer_name: "Adam" or "SGD".
        weight_decay: L2 regularization.
        task: "classification" or "regression".
        num_classes: Number of classes.
        weighted_by_class: Use class-weighted loss.
        es_metric: Early stopping metric.

    Returns:
        List of trained nn.Module models.
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    models = []
    for i, factory in enumerate(model_factories):
        print(f"[Pool] Training model {i + 1}/{len(model_factories)}...")
        model = factory()
        best_epoch, best_metric, model = fit_classifier(
            model, train_loader, val_loader, device,
            max_epochs=max_epochs, patience=patience, lr=lr,
            optimizer_name=optimizer_name, weight_decay=weight_decay,
            task=task, num_classes=num_classes,
            weighted_by_class=weighted_by_class, es_metric=es_metric,
        )
        print(f"[Pool] Model {i + 1}: best_epoch={best_epoch}, metric={best_metric:.4f}")
        model = model.cpu()
        models.append(model)

    return models


def train_pool_oof(
    model_factories: list[Callable[[], nn.Module]],
    train_dataset: Dataset,
    val_dataset: Dataset,
    device: torch.device,
    *,
    n_folds: int = 5,
    inner_val_ratio: float = 0.2,
    batch_size: int = 64,
    max_epochs: int = 300,
    patience: int = 20,
    lr: float = 0.0005,
    optimizer_name: str = "Adam",
    weight_decay: float = 5e-4,
    task: str = "classification",
    num_classes: int = 10,
    weighted_by_class: bool = True,
    es_metric: str = "val_loss",
    seed: int = 0,
    collate_fn: Callable | None = None,
) -> tuple[list[nn.Module], torch.Tensor, torch.Tensor]:
    """Train the pool with out-of-fold stacking, giving out-of-sample meta-labels.

    Each model is trained ``n_folds`` times. Within a fold, the K-1 fitting folds
    are split again into an inner train/val pair; early stopping uses the inner
    val, and the held-out fold is used only for prediction.

    After OOF collection, final models are retrained on all of ``train_dataset``
    for the fold-averaged best epoch count, early-stopping against
    ``val_dataset``.

    Args:
        model_factories: Callables each returning a fresh nn.Module.
        train_dataset: Data the pool is fit on. Folded; never includes val.
        val_dataset: Held-out validation set for the final retrain.
        n_folds: Number of outer CV folds.
        inner_val_ratio: Fraction of each fold's fitting data reserved for
            early stopping.
        seed: Seeds both the outer folds and the inner splits.

    Returns:
        (models, oof_logits, labels) -- ``oof_logits`` is [N, M, C] for
        classification or [N, M] for regression, every row predicted by a model
        that never saw that row.
    """
    N, M = len(train_dataset), len(model_factories)

    all_labels = []
    for i in range(N):
        _, y = train_dataset[i]
        value = y.item() if torch.is_tensor(y) else y
        all_labels.append(float(value) if task == "regression" else int(value))
    all_labels_arr = np.array(all_labels)
    labels_tensor = torch.tensor(
        all_labels_arr,
        dtype=torch.float if task == "regression" else torch.long,
    )

    oof_logits = (torch.zeros(N, M, num_classes) if task == "classification"
                  else torch.zeros(N, M))


    best_epochs_per_model = [[] for _ in range(M)]
    loader = lambda subset, shuffle: DataLoader(
        subset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

    if task == "regression":
        outer_splits = KFold(
            n_splits=n_folds, shuffle=True, random_state=seed).split(np.arange(N))
    else:
        outer_splits = StratifiedKFold(
            n_splits=n_folds, shuffle=True, random_state=seed,
        ).split(np.arange(N), all_labels_arr)

    for fold_idx, (fit_idx, oof_idx) in enumerate(outer_splits):
        # Inner split: early stopping never touches the held-out fold.
        if task == "regression":
            inner = ShuffleSplit(
                n_splits=1, test_size=inner_val_ratio,
                random_state=seed + fold_idx)
            rel_train, rel_val = next(inner.split(fit_idx))
        else:
            inner = StratifiedShuffleSplit(
                n_splits=1, test_size=inner_val_ratio,
                random_state=seed + fold_idx)
            rel_train, rel_val = next(
                inner.split(fit_idx, all_labels_arr[fit_idx]))
        inner_train_idx, inner_val_idx = fit_idx[rel_train], fit_idx[rel_val]

        inner_train_loader = loader(Subset(train_dataset, inner_train_idx.tolist()), True)
        inner_val_loader = loader(Subset(train_dataset, inner_val_idx.tolist()), False)
        oof_loader = loader(Subset(train_dataset, oof_idx.tolist()), False)

        print(f"[OOF] Fold {fold_idx + 1}/{n_folds} "
              f"(fit {len(inner_train_idx)}, inner-val {len(inner_val_idx)}, "
              f"held-out {len(oof_idx)})")

        for model_idx, factory in enumerate(model_factories):
            best_epoch, _, model = fit_classifier(
                factory(), inner_train_loader, inner_val_loader, device,
                max_epochs=max_epochs, patience=patience, lr=lr,
                optimizer_name=optimizer_name, weight_decay=weight_decay,
                task=task, num_classes=num_classes,
                weighted_by_class=weighted_by_class, es_metric=es_metric,
            )
            best_epochs_per_model[model_idx].append(best_epoch)

            preds = _collect_predictions(model, oof_loader, device, task=task)
            if task == "classification":
                oof_logits[oof_idx, model_idx, :] = preds
            else:
                oof_logits[oof_idx, model_idx] = preds
            model.cpu()
            del model            # fold models are discarded; only their predictions are kept

    # Final models: all of train_dataset, bounded by the fold-averaged budget.
    print("[OOF] Retraining final models on full training data...")
    full_loader = loader(train_dataset, True)
    val_loader = loader(val_dataset, False)

    models = []
    for model_idx, factory in enumerate(model_factories):
        avg_best = max(1, int(np.mean(best_epochs_per_model[model_idx])))
        print(f"[OOF] Retraining model {model_idx + 1}/{M} "
              f"for up to {avg_best} epochs...")
        _, _, model = fit_classifier(
            factory(), full_loader, val_loader, device,
            max_epochs=avg_best, patience=avg_best + 1, lr=lr,
            optimizer_name=optimizer_name, weight_decay=weight_decay,
            task=task, num_classes=num_classes,
            weighted_by_class=weighted_by_class, es_metric=es_metric,
        )
        model = model.cpu()
        models.append(model)

    return models, oof_logits, labels_tensor


# ── Prediction collection ──────────────────────────────────────────────

@torch.no_grad()
def _collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    task: str = "classification",
) -> torch.Tensor:
    """Collect raw model outputs over a data loader.

    Returns logits [N, C] for classification or predictions [N] for regression.
    """
    model.eval()
    all_out = []
    for x, _ in loader:
        x = x.to(device)
        out = model(x)
        if task == "regression":
            if out.ndim == 2 and out.size(1) == 1:
                out = out[:, 0]
            elif out.ndim != 1:
                raise ValueError(
                    "Scalar regression models must return [N] or [N, 1] outputs; "
                    f"got {tuple(out.shape)} from {type(model).__name__}.")
            all_out.append(out.cpu())
        else:
            all_out.append(out.cpu())
    return torch.cat(all_out, dim=0)


@torch.no_grad()
def collect_embeddings(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> torch.Tensor:
    """Penultimate activations for one model: whatever feeds its final Linear.

    Found by hooking the last ``nn.Linear`` in the module tree and capturing its
    input. A model that does not end in one raises.
    """
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    if not linears:
        raise ValueError(
            f"{type(model).__name__} has no nn.Linear, so it has no penultimate "
            "layer to read. Embedding feature sources need one.")
    captured: list[torch.Tensor] = []
    handle = linears[-1].register_forward_pre_hook(
        lambda _m, inputs: captured.append(inputs[0].detach().cpu()))
    try:
        model.eval()
        for x, _ in loader:
            model(x.to(device))
    finally:
        handle.remove()
    return torch.cat(captured, dim=0)


@torch.no_grad()
def collect_pool_embeddings(
    models: list[nn.Module],
    loader: DataLoader,
    device: torch.device,
) -> list[torch.Tensor]:
    """Penultimate activations from every pool member: M tensors of [N, D_m].

    Widths differ across a heterogeneous pool, so these are returned per model
    rather than stacked -- the caller decides whether to concatenate (any
    widths) or average (equal widths only).
    """
    out = []
    for model in models:
        model = model.to(device)
        out.append(collect_embeddings(model, loader, device))
        model.cpu()
    return out


@torch.no_grad()
def collect_pool_predictions(
    models: list[nn.Module],
    loader: DataLoader,
    device: torch.device,
    task: str = "classification",
) -> torch.Tensor:
    """Collect predictions from all pool models.

    Args:
        models: Trained pool models.
        loader: Data loader to run inference on.
        device: Compute device.
        task: "classification" or "regression".

    Returns:
        [N, M, C] logits for classification, or [N, M] for regression.
    """
    all_model_preds = []
    for model in models:
        model = model.to(device)
        preds = _collect_predictions(model, loader, device, task=task)
        model.cpu()
        all_model_preds.append(preds.unsqueeze(1))  # [N, 1, C] or [N, 1]

    return torch.cat(all_model_preds, dim=1)  # [N, M, C] or [N, M]


# ── Calibration ─────────────────────────────────────────────────────────

def calibrate_pool(
    logits: torch.Tensor,
    labels: torch.Tensor,
    calib_method: str = "ts-mix",
) -> tuple[torch.Tensor, list]:
    """Fit one calibrator per pool member and calibrate these logits with it.

    Each member is calibrated independently: they differ in architecture and in
    how overconfident they are, so a shared correction would under-correct one
    and over-correct another.

    Args:
        logits: Raw logits [N, M, C].
        labels: Ground truth labels [N].
        calib_method: Calibration method (default "ts-mix").

    Returns:
        (probs [N, M, C], calibrators list).
    """
    calibrators = [
        get_calibrator(calib_method, num_classes=logits.size(2))
        .fit(logits[:, i, :], labels)
        for i in range(logits.size(1))
    ]
    return apply_calibrators(calibrators, logits), calibrators


def apply_calibrators(
    calibrators: list,
    logits: torch.Tensor,
) -> torch.Tensor:
    """Apply pre-fitted calibrators to new logits [N, M, C].

    Returns calibrated probabilities [N, M, C].
    """
    probs_per_model = [calibrators[i].predict_proba(logits[:, i, :]).unsqueeze(1)
                       for i in range(logits.size(1))]
    return torch.cat(probs_per_model, dim=1)


# ── Data splitting ──────────────────────────────────────────────────────

def split_train_meta(
    dataset: Dataset,
    val_ratio: float = 0.5,
    seed: int = 0,
    task: str = "classification",
) -> tuple[Subset, Subset]:
    """Split dataset into base-training and meta-training halves.

    Used in split_train mode: base models train on one half and competence
    targets are generated from the other half.

    Args:
        dataset: Full training dataset.
        val_ratio: Fraction for the meta-training split (default 0.5).
        seed: Random seed.
        task: Classification uses stratification; regression uses a random split.

    Returns:
        (base_subset, meta_subset)
    """
    N = len(dataset)
    indices = np.arange(N)
    if task == "regression":
        splitter = ShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
        base_idx, meta_idx = next(splitter.split(indices))
    else:
        labels = []
        for i in range(N):
            _, y = dataset[i]
            labels.append(int(y.item()) if torch.is_tensor(y) else int(y))
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=val_ratio, random_state=seed)
        base_idx, meta_idx = next(splitter.split(indices, np.asarray(labels)))

    return Subset(dataset, base_idx.tolist()), Subset(dataset, meta_idx.tolist())
