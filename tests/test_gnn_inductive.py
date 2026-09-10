"""Prediction-composition invariants for GraphRoute GNN architectures."""
from __future__ import annotations

import pytest
import torch

from graphroute.gnn import HeteroGAT, SampleGAT, SampleGraphGPS, SampleMLP
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


def _graph(eval_indices: torch.Tensor, *, hetero: bool = False):
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
        include_classifier_context=hetero,
    )
    return data


def test_graph_build_detaches_tensor_features_before_numpy_conversion():
    """Graph construction accepts tensors that still track gradients."""
    train_features = TRAIN_FEATURES.clone().requires_grad_()
    eval_features = EVAL_FEATURES[:2].clone().requires_grad_()
    data, _ = build_graph(
        train_features,
        TRAIN_LABELS,
        TRAIN_DS,
        TRAIN_META,
        eval_features=eval_features,
        eval_labels=torch.zeros(2, dtype=torch.long),
        eval_ds=EVAL_DS[:2],
        train_edge_features=train_features,
        eval_edge_features=eval_features,
        k=2,
        neighbor_mode="knn",
        weight_mode="uniform",
        num_classes=NUM_CLASSES,
        eval_type="test",
    )

    assert data["sample"].x.shape == (len(TRAIN_FEATURES) + 2, FEATURE_DIM)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_graph_build_accepts_cuda_node_and_edge_features():
    """GPU inference may pass CUDA tensors into CPU-based graph construction."""
    device = torch.device("cuda")
    data, _ = build_graph(
        TRAIN_FEATURES.to(device),
        TRAIN_LABELS.to(device),
        TRAIN_DS.to(device),
        TRAIN_META.to(device),
        eval_features=EVAL_FEATURES[:2].to(device),
        eval_labels=torch.zeros(2, dtype=torch.long, device=device),
        eval_ds=EVAL_DS[:2].to(device),
        train_edge_features=TRAIN_FEATURES.to(device),
        eval_edge_features=EVAL_FEATURES[:2].to(device),
        k=2,
        neighbor_mode="knn",
        weight_mode="uniform",
        num_classes=NUM_CLASSES,
        eval_type="test",
    )

    assert data["sample"].x.device.type == "cpu"
    assert data["sample"].x.shape == (len(TRAIN_FEATURES) + 2, FEATURE_DIM)


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
    elif arch == "hetero_gat":
        model = HeteroGAT(
            **common, heads=2, attn_dropout=0.0, edge_dropout=0.0)
    else:
        model = SampleGraphGPS(
            **common, heads=2, attn_dropout=0.0, edge_dropout=0.0)
    return model.eval()


@pytest.mark.parametrize("arch", ["gat", "hetero_gat", "graph_gps", "mlp"])
def test_predictions_do_not_depend_on_evaluation_batch_composition(arch):
    """A complete split must equal any concatenation of its query batches."""
    torch.manual_seed(7)
    model = _model(arch)
    all_indices = torch.arange(len(EVAL_FEATURES))

    full_graph = _graph(all_indices, hetero=arch == "hetero_gat")
    full = model(full_graph)[full_graph["sample"].test_mask]

    parts = []
    for indices in (torch.tensor([0, 1]), torch.tensor([2]), torch.tensor([3, 4])):
        graph = _graph(indices, hetero=arch == "hetero_gat")
        parts.append(model(graph)[graph["sample"].test_mask])

    assert torch.allclose(full, torch.cat(parts), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("arch", ["gat", "hetero_gat", "graph_gps", "mlp"])
def test_reordering_evaluation_rows_only_reorders_predictions(arch):
    torch.manual_seed(11)
    model = _model(arch)
    original_indices = torch.arange(len(EVAL_FEATURES))
    permutation = torch.tensor([3, 0, 4, 1, 2])

    original_graph = _graph(original_indices, hetero=arch == "hetero_gat")
    original = model(original_graph)[original_graph["sample"].test_mask]
    permuted_graph = _graph(permutation, hetero=arch == "hetero_gat")
    permuted = model(permuted_graph)[permuted_graph["sample"].test_mask]

    assert torch.allclose(permuted, original[permutation], atol=1e-6, rtol=1e-6)


def test_hetero_gat_hides_a_training_query_correctness_from_itself():
    torch.manual_seed(17)
    model = _model("hetero_gat")
    first = model(_graph(torch.tensor([0]), hetero=True))

    original_meta = TRAIN_META[0].clone()
    try:
        TRAIN_META[0] = 1.0 - TRAIN_META[0]
        changed = model(_graph(torch.tensor([0]), hetero=True))
    finally:
        TRAIN_META[0].copy_(original_meta)

    assert torch.allclose(first[0], changed[0], atol=1e-6, rtol=1e-6)


def test_hetero_graph_uses_only_training_correctness_edges():
    graph = _graph(torch.tensor([0, 1]), hetero=True)
    correct_rel = ("classifier", "correct", "context")
    similar_rel = ("context", "similar", "sample")

    assert graph["context"].num_nodes == len(TRAIN_FEATURES)
    assert graph["classifier"].num_nodes == POOL_SIZE
    assert graph[correct_rel].edge_index.size(1) == int(TRAIN_META.sum().item())
    assert correct_rel in graph.edge_types
    assert similar_rel in graph.edge_types
    # A training row's context copy must never feed its own query copy.
    similar = graph[similar_rel].edge_index
    assert not bool((similar[0] == similar[1]).any())


@pytest.mark.parametrize("loss_target", ["meta_labels", "ensemble"])
def test_hetero_gat_runs_end_to_end_with_existing_objectives(loss_target):
    from torch.utils.data import TensorDataset

    from graphroute.config import GraphRouteConfig
    from graphroute.run import fit_graphroute

    torch.manual_seed(31)
    dataset = TensorDataset(
        torch.randn(32, FEATURE_DIM), torch.arange(32) % NUM_CLASSES)
    cfg = GraphRouteConfig(
        dataset="hetero-test", num_classes=NUM_CLASSES, device="cpu",
        loss_target=loss_target, val_ratio=0.25,
        base={"split_mode": "oof_stacking", "oof_folds": 2,
              "epochs": 1, "es_patience": 1, "batch_size": 8},
        graph={"pool_calibrate": False, "k": 2},
        gnn={"arch": "hetero_gat", "output_head": "dot",
             "epochs": 2, "patience": 1},
    )
    fitted = fit_graphroute(
        cfg, dataset, models=[torch.nn.Linear(FEATURE_DIM, NUM_CLASSES)],
        cache_dir="",
    )
    result = fitted.predict(dataset, cache_outputs=False)

    assert result["predictions"].shape == (len(dataset),)
    assert result["selection_scores"].shape == (len(dataset), 1)
    assert torch.isfinite(result["selection_scores"]).all()


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
