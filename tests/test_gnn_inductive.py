"""Prediction-composition invariants for GraphRoute GNN architectures."""
from __future__ import annotations

import pytest
import torch

from graphroute.gnn import SampleGAT, SampleGraphGPS, SampleMLP
from graphroute.graph import build_graph


FEATURE_DIM = 4
HIDDEN_DIM = 8
NUM_CLASSES = 2
POOL_SIZE = 3

_generator = torch.Generator().manual_seed(20260831)
TRAIN_FEATURES = torch.randn(6, FEATURE_DIM, generator=_generator)
TRAIN_LABELS = torch.tensor([0, 1, 0, 1, 0, 1])
TRAIN_DS = torch.randn(6, POOL_SIZE * NUM_CLASSES, generator=_generator)
TRAIN_META = torch.randint(0, 2, (6, POOL_SIZE), generator=_generator).float()
EVAL_FEATURES = torch.randn(5, FEATURE_DIM, generator=_generator)
EVAL_DS = torch.randn(5, POOL_SIZE * NUM_CLASSES, generator=_generator)


def _graph(eval_indices: torch.Tensor):
    eval_features = EVAL_FEATURES[eval_indices]
    eval_ds = EVAL_DS[eval_indices]
    data, _ = build_graph(
        TRAIN_FEATURES,
        TRAIN_LABELS,
        TRAIN_DS,
        TRAIN_META,
        eval_features=eval_features,
        # Evaluation labels are not model inputs. They are present only because
        # the generic graph container carries metric targets.
        eval_labels=torch.zeros(len(eval_indices), dtype=torch.long),
        eval_ds=eval_ds,
        train_edge_features=TRAIN_FEATURES,
        eval_edge_features=eval_features,
        k=2,
        neighbor_mode="knn",
        weight_mode="uniform",
        num_classes=NUM_CLASSES,
        eval_type="test",
    )
    return data


def _model(arch: str):
    common = dict(
        input_dim=FEATURE_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=2,
        feat_dropout=0.0,
        out_dim=POOL_SIZE,
    )
    if arch == "mlp":
        model = SampleMLP(**common)
    elif arch == "gat":
        model = SampleGAT(
            **common, heads=2, attn_dropout=0.0, edge_dropout=0.0)
    else:
        model = SampleGraphGPS(
            **common, heads=2, attn_dropout=0.0, edge_dropout=0.0)
    return model.eval()


@pytest.mark.parametrize("arch", ["gat", "graph_gps", "mlp"])
def test_predictions_do_not_depend_on_evaluation_batch_composition(arch):
    """A complete split must equal any concatenation of its query batches."""
    torch.manual_seed(7)
    model = _model(arch)
    all_indices = torch.arange(len(EVAL_FEATURES))

    full_graph = _graph(all_indices)
    full = model(full_graph)[full_graph["sample"].test_mask]

    parts = []
    for indices in (torch.tensor([0, 1]), torch.tensor([2]), torch.tensor([3, 4])):
        graph = _graph(indices)
        parts.append(model(graph)[graph["sample"].test_mask])

    assert torch.allclose(full, torch.cat(parts), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("arch", ["gat", "graph_gps", "mlp"])
def test_reordering_evaluation_rows_only_reorders_predictions(arch):
    torch.manual_seed(11)
    model = _model(arch)
    original_indices = torch.arange(len(EVAL_FEATURES))
    permutation = torch.tensor([3, 0, 4, 1, 2])

    original_graph = _graph(original_indices)
    original = model(original_graph)[original_graph["sample"].test_mask]
    permuted_graph = _graph(permutation)
    permuted = model(permuted_graph)[permuted_graph["sample"].test_mask]

    assert torch.allclose(permuted, original[permutation], atol=1e-6, rtol=1e-6)


def test_graph_gps_evaluation_nodes_cannot_change_other_nodes():
    """Changing one query cannot affect training nodes or another query."""
    torch.manual_seed(19)
    model = _model("graph_gps")
    base = _graph(torch.tensor([0, 1]))

    original_features = EVAL_FEATURES[1].clone()
    original_ds = EVAL_DS[1].clone()
    try:
        first = model(base)
        EVAL_FEATURES[1].add_(100.0)
        EVAL_DS[1].mul_(-50.0)
        changed = model(_graph(torch.tensor([0, 1])))
    finally:
        EVAL_FEATURES[1].copy_(original_features)
        EVAL_DS[1].copy_(original_ds)

    # Six training rows and the first evaluation row must be unchanged.
    assert torch.allclose(first[:7], changed[:7], atol=1e-6, rtol=1e-6)


def test_graph_gps_training_gradients_cannot_depend_on_validation_features():
    """Validation features must not participate in fitting the GNN."""
    torch.manual_seed(23)
    model = _model("graph_gps").train()

    def gradients(graph):
        model.zero_grad(set_to_none=True)
        train_scores = model(graph)[graph["sample"].train_mask]
        train_scores.square().mean().backward()
        return [parameter.grad.detach().clone() for parameter in model.parameters()]

    base_gradients = gradients(_graph(torch.tensor([0, 1, 2])))
    original_features = EVAL_FEATURES[:3].clone()
    original_ds = EVAL_DS[:3].clone()
    try:
        EVAL_FEATURES[:3].normal_(mean=500.0, std=100.0)
        EVAL_DS[:3].normal_(mean=-500.0, std=100.0)
        changed_gradients = gradients(_graph(torch.tensor([0, 1, 2])))
    finally:
        EVAL_FEATURES[:3].copy_(original_features)
        EVAL_DS[:3].copy_(original_ds)

    for before, after in zip(base_gradients, changed_gradients):
        assert torch.allclose(before, after, atol=1e-6, rtol=1e-6)


def test_graph_gps_requires_an_explicit_nonempty_training_memory():
    torch.manual_seed(29)
    model = _model("graph_gps")
    graph = _graph(torch.tensor([0]))

    del graph["sample"].train_mask
    with pytest.raises(ValueError, match="train_mask"):
        model(graph)

    graph = _graph(torch.tensor([0]))
    graph["sample"].train_mask.zero_()
    with pytest.raises(ValueError, match="at least one training node"):
        model(graph)
