"""Voting-weight transformations and ensemble aggregation."""

import torch

from graphroute.selection import (
    compute_selection_matrix,
    evaluate_ensemble,
    evaluate_ensemble_regression,
)


def test_dense_uses_every_sigmoid_weight_without_thresholding():
    logits = torch.tensor([[-2.0, 0.0, 2.0]])

    selection, fallback = compute_selection_matrix(
        logits, "soft_weighted_voting", "dense"
    )

    assert torch.allclose(selection, torch.sigmoid(logits))
    assert torch.all(selection > 0)
    assert fallback.tolist() == [False]

    thresholded, _ = compute_selection_matrix(
        logits, "soft_weighted_voting", "sig"
    )
    assert thresholded[0, :2].tolist() == [0.0, 0.0]
    assert thresholded[0, 2] > 0


def test_dense_soft_weighted_voting_matches_ensemble_training_weights():
    logits = torch.tensor([[-2.0, 2.0]])
    decision_space = torch.tensor([[0.9, 0.1, 0.2, 0.8]])
    weights = torch.sigmoid(logits)
    expected = (
        weights[0, 0] * decision_space[0, :2]
        + weights[0, 1] * decision_space[0, 2:]
    ) / weights.sum()

    probabilities, predictions = evaluate_ensemble(
        logits,
        decision_space,
        num_classes=2,
        combination_mode="soft_weighted_voting",
        voting_weight_space="dense",
    )

    assert torch.allclose(probabilities[0], expected)
    assert predictions.item() == expected.argmax().item()


def test_dense_regression_uses_the_same_unthresholded_weights():
    logits = torch.tensor([[-2.0, 2.0]])
    predictions = torch.tensor([[1.0, 3.0]])
    weights = torch.sigmoid(logits)
    expected = (weights * predictions).sum(dim=1) / weights.sum(dim=1)

    actual = evaluate_ensemble_regression(logits, predictions, "dense")

    assert torch.allclose(actual, expected)
