"""Loss functions and meta-label computation for GraphRoute."""
from __future__ import annotations

import torch
import torch.nn.functional as F


# ── Meta-label computation ───────────────────────────────────────────────

def compute_meta_labels(
    preds: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Binary correctness meta-labels: was classifier j right about sample i?

    This is what the GNN learns to predict. A row of all zeros -- no classifier
    got that sample right -- is left as is; the ensemble's fallback handles those
    at inference rather than the labels pretending some classifier was competent.

    Args:
        preds: Hard predictions [N, M].
        labels: True labels [N].

    Returns:
        Binary meta-label tensor [N, M].
    """
    return (preds == labels.unsqueeze(1)).float()


def compute_regression_meta_labels(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    scale: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """Continuous competence targets for scalar regression.

    ``competence[i, j] = exp(-|pred_j - y_i| / scale)``

    ``scale`` is fitted once from the meta-training targets and then reused for
    validation so a given error has the same meaning on both splits.

    Args:
        predictions: Regressor predictions [N, M].
        targets: True target values [N].
        scale: Positive target scale. Defaults to the population standard
            deviation of ``targets`` when the function is used independently.

    Returns:
        Continuous competence targets in (0, 1], shape [N, M].
    """
    targets = targets.float().reshape(-1)
    if scale is None:
        scale_t = targets.std(correction=0)
    else:
        scale_t = torch.as_tensor(scale, dtype=targets.dtype, device=targets.device)
    if not torch.isfinite(scale_t) or scale_t <= 0:
        raise ValueError(
            "Regression competence needs a positive finite target scale; the "
            "meta-training targets have no usable variation.")
    abs_err = (predictions.float() - targets.unsqueeze(1)).abs()
    return torch.exp(-abs_err / scale_t)


# ── Margin targets (for soft_bce / regression loss modes) ────────────────

def margin_targets(ds: torch.Tensor, y: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Per-classifier margin = P(true class) - max P(other class).

    Args:
        ds: Decision-space tensor [N, M*C] (flattened probabilities).
        y: True labels [N].
        num_classes: Number of classes C.

    Returns:
        Tensor [N, M] of margin values in [-1, 1].
    """
    M = int(ds.size(1) // max(num_classes, 1))
    probs = ds.view(ds.size(0), M, num_classes).float()
    y_idx = y.view(-1, 1, 1).expand(-1, M, 1)
    p_true = probs.gather(2, y_idx).squeeze(-1)
    probs_other = probs.clone()
    probs_other.scatter_(2, y_idx, -1.0)
    p_other = probs_other.max(dim=2).values
    return p_true - p_other


# ── Meta-label loss ──────────────────────────────────────────────────────

def compute_meta_loss(
    logits: torch.Tensor,
    train_meta: torch.Tensor,
    gnn_loss_mode: str,
    sample_weights: torch.Tensor | None = None,
    train_margin: torch.Tensor | None = None,
    seed_local: torch.Tensor | None = None,
    focal_gamma: float = 2.0,
    meta_min_pos: int = 1,
) -> torch.Tensor:
    """Compute meta-label loss for GNN training.

    Args:
        logits: GNN output logits [B, M].
        train_meta: Binary correctness or continuous competence targets [B, M].
        gnn_loss_mode: One of "bce", "focal_bce", "soft_bce", "regression".
        sample_weights: Optional per-sample weights [N] or [B].
        train_margin: Precomputed margin targets [N, M] (for regression/soft_bce).
        seed_local: Mini-batch index mapping into full tensors.
        focal_gamma: Focusing parameter for focal_bce mode.
        meta_min_pos: Min positives for soft_bce boost.

    Returns:
        Scalar loss tensor.
    """
    criterion_bce = torch.nn.BCEWithLogitsLoss(reduction="none")
    criterion_mse = torch.nn.MSELoss(reduction="none")

    # Index into full tensors if mini-batch
    # Mini-batch: index into the full-graph tensors. Only the margins are needed
    # here -- the loss is computed against meta-labels, not the decision space.
    if seed_local is not None and train_margin is not None:
        batch_margin = train_margin[seed_local]
    else:
        batch_margin = train_margin

    if gnn_loss_mode == "regression":
        # For classification: use margin targets; for regression task: use
        # continuous competence targets directly (already in [0, 1]).
        if batch_margin is not None:
            pred_for_loss = torch.tanh(logits)
            target = batch_margin
        else:
            pred_for_loss = torch.sigmoid(logits)
            target = train_meta
        per_elem = criterion_mse(pred_for_loss, target)
        per_sample = per_elem.mean(dim=1)

    elif gnn_loss_mode == "soft_bce":
        raw_margin = batch_margin
        if meta_min_pos > 0:
            topk_vals, _ = torch.topk(
                raw_margin,
                k=min(meta_min_pos, raw_margin.size(1)),
                dim=1,
            )
            thresholds = topk_vals[:, -1].unsqueeze(1)
            boost_val = 0.1
            raw_margin = torch.where(
                (raw_margin >= thresholds) & (raw_margin < boost_val),
                raw_margin.new_tensor(boost_val),
                raw_margin,
            )
        soft_target = (raw_margin + 1.0) / 2.0
        per_elem = criterion_bce(logits, soft_target)
        per_sample = per_elem.mean(dim=1)

    elif gnn_loss_mode == "focal_bce":
        per_elem = criterion_bce(logits, train_meta)
        with torch.no_grad():
            p = torch.sigmoid(logits)
            p_t = train_meta * p + (1.0 - train_meta) * (1.0 - p)
            focal_weight = (1.0 - p_t).pow(focal_gamma)
        per_elem = per_elem * focal_weight
        per_sample = per_elem.mean(dim=1)

    else:  # bce (default)
        per_elem = criterion_bce(logits, train_meta)
        per_sample = per_elem.mean(dim=1)

    if sample_weights is not None:
        sw = sample_weights if seed_local is None else sample_weights[seed_local]
        per_sample = per_sample * sw.to(per_sample.device)

    return per_sample.mean()


# ── Ensemble loss (end-to-end) ───────────────────────────────────────────

def compute_ensemble_loss(
    logits: torch.Tensor,
    ds: torch.Tensor,
    y: torch.Tensor,
    num_classes: int,
    task: str = "classification",
    sample_weights: torch.Tensor | None = None,
    seed_local: torch.Tensor | None = None,
) -> torch.Tensor:
    """End-to-end ensemble loss: backprop through GNN weights to prediction.

    For classification: cross-entropy on the weighted ensemble probability.
    For regression: MSE on the weighted ensemble prediction.

    Args:
        logits: GNN output logits [B, M].
        ds: Base model outputs — [N, M*C] for classification or [N, M] for regression.
        y: True labels [N] (long for classification, float for regression).
        num_classes: Number of classes C (1 for regression).
        task: "classification" or "regression".
        sample_weights: Optional per-sample weights [N] or [B].
        seed_local: Mini-batch index mapping into full tensors.

    Returns:
        Scalar loss tensor.
    """
    if seed_local is not None:
        batch_ds = ds[seed_local]
        batch_y = y[seed_local]
    else:
        batch_ds = ds
        batch_y = y

    # GNN logits -> independent non-negative pool-member weights.
    w = torch.sigmoid(logits)  # [B, M]
    w_sum = w.sum(dim=1, keepdim=True).clamp(min=1e-8)

    if task == "classification":
        M = logits.size(1)
        p = batch_ds.view(-1, M, num_classes).float()  # [B, M, C]
        # Weighted average of class probabilities
        p_ensemble = (w.unsqueeze(-1) * p).sum(dim=1) / w_sum  # [B, C]
        # Cross-entropy loss
        log_p = torch.log(p_ensemble.clamp(min=1e-8))
        per_sample = F.nll_loss(log_p, batch_y, reduction="none")
    else:  # regression
        preds = batch_ds.float()  # [B, M]
        y_hat = (w * preds).sum(dim=1) / w_sum.squeeze(1)  # [B]
        per_sample = F.mse_loss(y_hat, batch_y.float(), reduction="none")

    if sample_weights is not None:
        sw = sample_weights if seed_local is None else sample_weights[seed_local]
        per_sample = per_sample * sw.to(per_sample.device)

    return per_sample.mean()
