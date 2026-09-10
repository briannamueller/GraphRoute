"""Feature sources decide what the graph is built on, so each must produce a
distinct representation -- five names that silently return the same array would
look like a working ablation and be nothing of the kind.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from graphroute.features import extract_feature_slice
from graphroute.graph import build_cmdw_edges, build_knn_edges
from graphroute.run import _inputs, build_features

N, M, C, D = 10, 3, 4, 7


def _middle_features(inputs):
    return inputs[:, 1:3]


@pytest.fixture
def parts():
    probs = torch.rand(N, M, C)
    probs /= probs.sum(-1, keepdim=True)
    return probs, torch.randn(N, D), [torch.randn(N, w) for w in (5, 8, 5)]


def test_each_source_has_its_own_width(parts):
    probs, raw, emb = parts
    widths = {src: build_features(src, probs, raw, emb).shape
              for src in ("decision_space", "feature_space", "hybrid", "embedding_concat")}
    assert widths["decision_space"] == (N, M * C)
    assert widths["feature_space"] == (N, D)
    assert widths["hybrid"] == (N, M * C + D)
    assert widths["embedding_concat"] == (N, 5 + 8 + 5)


def test_regression_predictions_are_already_decision_space():
    predictions = torch.randn(N, M)
    raw = torch.randn(N, D)
    assert torch.equal(
        build_features("decision_space", predictions, raw, None), predictions)
    assert build_features("hybrid", predictions, raw, None).shape == (N, M + D)


def test_sources_are_not_secretly_the_same_array(parts):
    probs, raw, emb = parts
    out = [build_features(s, probs, raw, emb).flatten()[:5]
           for s in ("decision_space", "feature_space", "hybrid")]
    for a, b in zip(out, out[1:]):
        assert not torch.allclose(a, b)


def test_embedding_mean_needs_matching_widths(parts):
    probs, raw, _ = parts
    same = [torch.randn(N, 6) for _ in range(M)]
    assert build_features("embedding_mean", probs, raw, same).shape == (N, 6)
    with pytest.raises(ValueError, match="widths must match"):
        build_features("embedding_mean", probs, raw, [torch.randn(N, w) for w in (6, 9, 6)])


def test_each_model_embedding_is_normalized_before_concatenation(parts):
    probs, raw, _ = parts
    embeddings = [torch.tensor([[3.0, 4.0]]).repeat(N, 1),
                  torch.tensor([[0.0, 0.0, 12.0]]).repeat(N, 1)]

    features = build_features(
        "embedding_concat", probs, raw, embeddings,
        embedding_normalization="per_model_l2",
    )

    assert torch.allclose(features[:, :2].norm(dim=1), torch.ones(N))
    assert torch.allclose(features[:, 2:].norm(dim=1), torch.ones(N))


def test_custom_feature_source_uses_supplied_features(parts):
    probs, raw, emb = parts
    custom = torch.randn(N, 3)
    assert torch.equal(
        build_features("diagnoses", probs, raw, emb, custom,
                       custom_source="diagnoses"),
        custom,
    )


def test_custom_feature_extractor_is_reused_for_prediction():
    from torch.utils.data import TensorDataset

    from graphroute.config import GraphRouteConfig
    from graphroute.run import fit_graphroute

    dataset = TensorDataset(torch.randn(32, 4), torch.arange(32) % 2)
    extractor = _middle_features
    cfg = GraphRouteConfig(
        dataset="custom-features",
        num_classes=2,
        device="cpu",
        base={"split_mode": "split_train", "epochs": 1, "batch_size": 8},
        graph={
            "node_feature_source": "clinical_features",
            "edge_feature_source": "clinical_features",
            "pool_calibrate": False,
            "k": 2,
        },
        gnn={"arch": "mlp", "epochs": 1, "patience": 1},
    )

    fitted = fit_graphroute(
        cfg,
        dataset,
        validation_set=dataset,
        models=[torch.nn.Linear(4, 2)],
        cache_dir="",
        feature_extractor=extractor,
    )
    predicted = fitted.predict(dataset, cache_outputs=False)

    assert fitted.train_node_features.shape[1] == 2
    assert fitted.feature_extractor is extractor
    assert predicted["predictions"].shape == (len(dataset),)


def test_custom_feature_source_requires_an_extractor():
    from torch.utils.data import TensorDataset

    from graphroute.config import GraphRouteConfig
    from graphroute.run import fit_graphroute

    cfg = GraphRouteConfig(
        dataset="custom-features",
        num_classes=2,
        graph={"edge_feature_source": "clinical_features"},
    )
    dataset = TensorDataset(torch.randn(8, 4), torch.arange(8) % 2)

    with pytest.raises(ValueError, match="requires feature_extractor"):
        fit_graphroute(cfg, dataset, cache_dir="")


def test_feature_slice_selects_one_input_field():
    inputs = (torch.randn(4, 3, 2), torch.arange(24).reshape(4, 6))
    selected = extract_feature_slice(inputs, input_index=1, start=2, stop=5)
    assert torch.equal(selected, inputs[1][:, 2:5])


def test_cosine_and_manhattan_can_select_different_neighbors():
    features = torch.tensor([
        [1.0, 0.0],
        [0.9, 0.1],
        [10.0, 0.0],
    ]).numpy()
    labels = torch.zeros(3, dtype=torch.long).numpy()
    sources = torch.tensor([1, 2]).numpy()
    destinations = torch.tensor([0]).numpy()

    manhattan, _ = build_knn_edges(
        features, labels, sources, destinations, k=1,
        weight_mode="uniform", distance_metric="manhattan")
    cosine, _ = build_knn_edges(
        features, labels, sources, destinations, k=1,
        weight_mode="uniform", distance_metric="cosine")
    cmdw, _ = build_cmdw_edges(
        features, labels, sources, destinations, k=1,
        neighbor_mode="class_balanced", distance_metric="cosine")

    assert manhattan[0].tolist() == [1]
    assert cosine[0].tolist() == [2]
    assert cmdw[0].tolist() == [2]


def test_missing_inputs_are_refused_not_guessed(parts):
    probs, _, emb = parts
    for src in ("feature_space", "hybrid"):
        with pytest.raises(ValueError):
            build_features(src, probs, None, emb)
    with pytest.raises(ValueError):
        build_features("embedding_mean", probs, None, None)
    with pytest.raises(ValueError, match="Unknown feature source"):
        build_features("nonsense", probs, None, None)


def test_feature_space_flattens_and_concatenates_multi_input_fields():
    ts = torch.arange(2 * 3 * 4).reshape(2, 3, 4)
    static = torch.arange(2 * 5).reshape(2, 5) + 100
    loader = [((ts, static), torch.tensor([0, 1]))]

    raw = _inputs(loader)

    assert raw.shape == (2, 17)
    assert torch.equal(raw[:, :12], ts.flatten(1).float())
    assert torch.equal(raw[:, 12:], static.float())


def test_binary_loss_is_bce_on_the_positive_logit():
    """Column 0 gets no gradient; column 1 carries the binary decision."""
    import torch.nn as nn

    from graphroute.pool import build_loss_fn
    model = nn.Linear(4, 2)
    build_loss_fn(2, False, torch.device("cpu"))(
        model(torch.randn(16, 4)), torch.randint(0, 2, (16,))).backward()
    assert model.weight.grad[0].norm() == 0        # column 0: untrained, by design
    assert model.weight.grad[1].norm() > 0


def test_balanced_accuracy_is_not_plain_accuracy():
    """val_bacc returned negative loss; a majority-only classifier must not score
    well on a metric named balanced accuracy."""
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    from graphroute.pool import build_loss_fn, evaluate

    class AlwaysZero(nn.Module):
        def forward(self, x):
            return torch.tensor([[10.0, -10.0]]).repeat(len(x), 1)

    y = torch.cat([torch.zeros(90, dtype=torch.long), torch.ones(10, dtype=torch.long)])
    stats = evaluate(AlwaysZero(),
                     DataLoader(TensorDataset(torch.randn(100, 4), y), batch_size=32),
                     torch.device("cpu"), build_loss_fn(2, False, torch.device("cpu")))
    assert abs(stats["acc"] - 0.90) < 1e-6
    assert abs(stats["bacc"] - 0.50) < 1e-6


def _tiny_graph(device):
    from torch_geometric.data import HeteroData
    d = HeteroData()
    d["sample"].x = torch.zeros(5, 1, device=device)
    d["sample"].train_mask = torch.tensor([True, True, False, True, True], device=device)
    rel = ("sample", "ss", "sample")
    d[rel].edge_index = torch.stack([torch.tensor([0, 1, 3, 4], device=device),
                                     torch.tensor([2, 2, 2, 2], device=device)])
    d[rel].edge_attr = torch.ones(4, device=device)
    return d, torch.tensor([False, False, True, False, False], device=device)


@pytest.mark.parametrize("mode,expected", [("acc", [0.75, 0.25]),
                                           ("wacc", [0.75, 0.25]),
                                           ("bacc", [0.5, 0.5])])
def test_fallback_modes(mode, expected):
    """acc/wacc let the neighbourhood's majority class decide; bacc must not."""
    from graphroute.training import FallbackModel
    d, mask = _tiny_graph(torch.device("cpu"))
    meta = torch.tensor([[1., 0.], [1., 0.], [1., 0.], [0., 1.]])   # train rows
    got = FallbackModel(mode, meta, torch.tensor([0, 0, 0, 1]))(d, mask).flatten()
    assert torch.allclose(got, torch.tensor(expected), atol=1e-6)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no accelerator")
def test_fallback_runs_off_cpu():
    """Fallback computations keep their outputs on the graph device."""
    from graphroute.training import FallbackModel, _neighborhood_gain
    dev = torch.device("mps")
    d, mask = _tiny_graph(dev)
    meta = torch.tensor([[1., 0.], [1., 0.], [1., 0.], [0., 1.]])   # built on CPU
    for mode in ("acc", "wacc", "bacc"):
        out = FallbackModel(mode, meta, torch.tensor([0, 0, 0, 1]))(d, mask)
        assert out.device.type == "mps"
    assert _neighborhood_gain(d, torch.zeros(5, 2, device=dev)).device.type == "mps"


@pytest.mark.parametrize("source", ["embedding_mean", "embedding_concat"])
def test_oof_predictions_and_final_model_embeddings_are_separate_channels(source):
    from torch.utils.data import TensorDataset

    from graphroute.config import GraphRouteConfig
    from graphroute.run import fit_graphroute
    cfg = GraphRouteConfig(
        dataset="test", num_classes=2, device="cpu",
        base={"models": ["mlp8"], "split_mode": "oof_stacking",
              "oof_folds": 2, "epochs": 1, "batch_size": 8},
        graph={"node_feature_source": source, "edge_feature_source": source,
               "pool_calibrate": False, "k": 2},
        gnn={"arch": "mlp", "epochs": 1, "patience": 1})
    ds = TensorDataset(torch.randn(40, 4), torch.arange(40) % 2)
    fitted = fit_graphroute(cfg, ds, cache_dir="")

    n_train = len(fitted.train_labels)
    assert fitted.train_node_features.shape == (n_train, 8)
    assert fitted.train_edge_features.shape == (n_train, 8)
    assert fitted.train_decision_space.shape == (n_train, 2)


@pytest.mark.parametrize("task,labels,ds,num_classes", [
    ("classification", torch.randint(0, 3, (12,)), torch.rand(12, 9), 3),
    ("regression", torch.rand(12), torch.rand(12, 3), 1),
])
@pytest.mark.parametrize("mode", ["none", "class_prevalence", "difficulty"])
def test_every_sample_weight_mode_runs(mode, task, labels, ds, num_classes):
    """Every applicable weighting mode returns finite sample weights."""
    from graphroute.training import compute_sample_weights
    w = compute_sample_weights(mode, labels, ds, num_classes, task=task)
    if mode == "none" or (mode == "class_prevalence" and task == "regression"):
        assert w is None
    else:
        assert w.shape == (12,) and torch.isfinite(w).all()
