"""Base-model optimizer configuration reaches PyTorch unchanged."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from graphroute.pool import fit_classifier


def test_fit_classifier_uses_configured_sgd_momentum(monkeypatch):
    captured = {}
    real_sgd = torch.optim.SGD

    def recording_sgd(parameters, **kwargs):
        captured.update(kwargs)
        return real_sgd(parameters, **kwargs)

    monkeypatch.setattr(torch.optim, "SGD", recording_sgd)
    dataset = TensorDataset(torch.randn(8, 3), torch.tensor([0, 1] * 4))
    loader = DataLoader(dataset, batch_size=4)

    fit_classifier(
        nn.Linear(3, 2),
        loader,
        loader,
        torch.device("cpu"),
        max_epochs=1,
        patience=1,
        optimizer_name="SGD",
        momentum=0,
        num_classes=2,
        weighted_by_class=False,
    )

    assert captured["momentum"] == 0
