import math

import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset

from graphroute.config import GraphRouteConfig
from graphroute.losses import compute_regression_meta_labels
from graphroute.pool import split_train_meta
from graphroute.run import fit_graphroute


def _factory(width):
    return lambda: nn.Sequential(nn.Linear(3, width), nn.ReLU(), nn.Linear(width, 1))


def _datasets():
    generator = torch.Generator().manual_seed(4)
    x = torch.randn(72, 3, generator=generator)
    y = 1.5 * x[:, 0] - 0.7 * x[:, 1] + 0.2 * x[:, 2]
    y = y + 0.05 * torch.randn(72, generator=generator)
    return TensorDataset(x[:56], y[:56]), TensorDataset(x[56:], y[56:])


def test_regression_competence_is_smooth_and_uses_the_supplied_scale():
    targets = torch.tensor([-1.0, 1.0])
    predictions = torch.tensor([[-1.0, 0.0], [3.0, 1.0]])
    got = compute_regression_meta_labels(predictions, targets, scale=2.0)
    expected = torch.tensor([
        [1.0, math.exp(-0.5)],
        [math.exp(-1.0), 1.0],
    ])
    assert torch.allclose(got, expected)

    # A supplied training scale, rather than this split's target spread, controls
    # the meaning of an error on validation.
    shifted_targets = torch.tensor([-100.0, 100.0])
    shifted_predictions = shifted_targets[:, None] + 2.0
    shifted = compute_regression_meta_labels(
        shifted_predictions, shifted_targets, scale=2.0)
    assert torch.allclose(shifted, torch.full((2, 1), math.exp(-1.0)))


def test_regression_competence_refuses_a_constant_target_scale():
    with pytest.raises(ValueError, match="no usable variation"):
        compute_regression_meta_labels(torch.ones(4, 2), torch.ones(4))


def test_regression_split_does_not_turn_targets_into_classes():
    targets = torch.linspace(-1.0, 1.0, 24)
    dataset = TensorDataset(torch.randn(24, 3), targets)
    train, meta = split_train_meta(dataset, seed=2, task="regression")
    assert len(train) == len(meta) == 12


@pytest.mark.parametrize("split_mode", ["split_train", "oof_stacking"])
@pytest.mark.parametrize("loss_target", ["meta_labels", "ensemble"])
def test_scalar_regression_runs_end_to_end(split_mode, loss_target):
    train, test = _datasets()
    cfg = GraphRouteConfig(
        dataset="test-regression",
        task="regression",
        loss_target=loss_target,
        device="cpu",
        base={
            "split_mode": split_mode,
            "oof_folds": 2,
            "epochs": 2,
            "es_patience": 2,
            "batch_size": 16,
        },
        graph={"k": 2},
        gnn={
            "arch": "mlp",
            "hidden_dim": 8,
            "layers": 1,
            "epochs": 2,
            "patience": 2,
            "feat_dropout": 0.0,
            "attn_dropout": 0.0,
        },
    )
    model = fit_graphroute(
        cfg,
        train,
        models=[_factory(5)(), _factory(7)()],
        cache_dir="",
    )
    result = model.evaluate(test)

    assert math.isfinite(result["mse"])
    assert math.isfinite(result["rmse"])
    assert result["effective_pool_size"] >= 0
    assert "val_rmse" in model.history["history"][0]
    assert "val_acc" not in model.history["history"][0]
    if loss_target == "meta_labels":
        assert model.competence_scale > 0
    else:
        assert model.competence_scale is None
