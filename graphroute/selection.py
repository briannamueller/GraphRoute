"""Ensemble evaluation, selection, and inference for GraphRoute."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def evaluate_ensemble(
    logits: torch.Tensor,
    ds: torch.Tensor,
    num_classes: int,
    combination_mode: str,
    voting_weight_space: str = "logit",
    hard_preds: torch.Tensor | None = None,
    threshold: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aggregate base classifier outputs into ensemble predictions.

    Args:
        logits: GNN output logits [N, M].
        ds: Decision-space tensor [N, M*C] (classification) or [N, M] (regression).
        num_classes: Number of classes C.
        combination_mode: One of soft_weighted_voting, hard_weighted_voting,
            soft_voting, hard_voting.
        voting_weight_space: "logit" or "sig".
        hard_preds: Hard predictions [N, M] (required for hard voting modes).
        threshold: Optional sigmoid threshold override for selection.

    Returns:
        (soft_probs, hard_preds_out): [N, C] and [N] for classification.
    """
    M = logits.size(1)
    C = num_classes
    ds_local = ds.view(-1, M, C)

    if threshold is not None:
        sig = torch.sigmoid(logits)
        raw_weights = torch.where(sig > threshold, sig, torch.zeros_like(sig))
    elif "weighted" in combination_mode:
        if voting_weight_space == "sig":
            sig = torch.sigmoid(logits)
            raw_weights = torch.where(sig > 0.5, sig, torch.zeros_like(sig))
        else:  # "logit" (default)
            raw_weights = F.relu(logits)
    else:
        raw_weights = (logits > 0).float()

    # Fallback: if no classifier selected, weight all equally
    sum_w = raw_weights.sum(dim=1, keepdim=True)
    fallback_mask = (sum_w == 0)
    final_weights = torch.where(fallback_mask, torch.ones_like(raw_weights), raw_weights)
    sum_w_final = final_weights.sum(dim=1, keepdim=True).clamp(min=1e-6)
    norm_weights = final_weights / sum_w_final

    if "soft" in combination_mode:
        target = ds_local
    else:
        if hard_preds is None:
            raise ValueError(f"hard_preds required for '{combination_mode}' mode.")
        target = F.one_hot(hard_preds, num_classes=C).float()

    soft_probs = (norm_weights.unsqueeze(-1) * target).sum(dim=1)
    hard_out = soft_probs.argmax(dim=1)
    return soft_probs, hard_out


def evaluate_ensemble_regression(
    logits: torch.Tensor,
    predictions: torch.Tensor,
    voting_weight_space: str = "logit",
    threshold: float | None = None,
) -> torch.Tensor:
    """Aggregate regressor outputs into ensemble prediction.

    Args:
        logits: GNN output logits [N, M].
        predictions: Regressor predictions [N, M].
        voting_weight_space: "logit" or "sig".
        threshold: Optional sigmoid threshold override.

    Returns:
        Ensemble predictions [N].
    """
    if threshold is not None:
        sig = torch.sigmoid(logits)
        raw_weights = torch.where(sig > threshold, sig, torch.zeros_like(sig))
    elif voting_weight_space == "sig":
        sig = torch.sigmoid(logits)
        raw_weights = torch.where(sig > 0.5, sig, torch.zeros_like(sig))
    else:  # "logit"
        raw_weights = F.relu(logits)

    sum_w = raw_weights.sum(dim=1, keepdim=True)
    fallback_mask = (sum_w == 0)
    final_weights = torch.where(fallback_mask, torch.ones_like(raw_weights), raw_weights)
    sum_w_final = final_weights.sum(dim=1, keepdim=True).clamp(min=1e-6)
    norm_weights = final_weights / sum_w_final

    return (norm_weights * predictions.float()).sum(dim=1)


def compute_selection_matrix(
    logits: torch.Tensor,
    combination_mode: str,
    voting_weight_space: str = "logit",
    threshold: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute selection weights and fallback mask from GNN logits.

    Args:
        logits: GNN output logits [N, M].
        combination_mode: Aggregation mode string.
        voting_weight_space: "logit" or "sig".
        threshold: Optional sigmoid threshold override.

    Returns:
        (selection_matrix [N, M], fallback_rows [N] bool mask).
    """
    if threshold is not None:
        sig = torch.sigmoid(logits)
        selection_matrix = torch.where(sig > threshold, sig, torch.zeros_like(sig))
    elif "weighted" in combination_mode:
        if voting_weight_space == "sig":
            sig = torch.sigmoid(logits)
            selection_matrix = torch.where(sig > 0.5, sig, torch.zeros_like(sig))
        else:  # "logit"
            selection_matrix = F.relu(logits)
    else:
        selection_matrix = (logits > 0).float()

    fallback_mask = selection_matrix.sum(dim=1, keepdim=True) == 0
    if fallback_mask.any():
        selection_matrix = torch.where(
            fallback_mask, torch.ones_like(selection_matrix), selection_matrix
        )
    fallback_rows = fallback_mask.squeeze(1)

    return selection_matrix, fallback_rows
