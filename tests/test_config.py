"""The config is only useful if it stays in step with the CLI and with the
functions it feeds. These pin the three seams where that could drift.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from graphroute.cli import build_parser, config_from_args
from graphroute.config import BaseConfig, GNNConfig, GraphConfig, GraphRouteConfig


def test_every_config_field_has_a_generated_flag():
    """Adding a Pydantic field automatically makes it available in the terminal."""
    expected = {
        name
        for name in GraphRouteConfig.model_fields
        if name not in {"base", "graph", "gnn"}
    }
    for group, model in (
        ("base", BaseConfig),
        ("graph", GraphConfig),
        ("gnn", GNNConfig),
    ):
        expected |= {f"{group}__{name}" for name in model.model_fields}

    actual = {
        action.dest
        for action in build_parser()._actions
        if action.dest != "help"
    }
    assert actual == expected


def test_flags_actually_reach_the_config():
    cfg = config_from_args(["--dataset", "test", "--graph-k", "9", "--gnn-hidden-dim", "256",
                            "--base-epochs", "7", "--gnn-arch", "mlp"])
    assert (cfg.graph.k, cfg.gnn.hidden_dim, cfg.base.epochs, cfg.gnn.arch) == (9, 256, 7, "mlp")


def test_unset_flags_keep_config_defaults():
    """Flags left off the command line must not overwrite defaults with None."""
    expected = GraphRouteConfig(dataset="test")
    assert config_from_args(["--dataset", "test"]).model_dump() == expected.model_dump()


def test_dataset_is_required_and_data_dir_has_a_default():
    with pytest.raises(SystemExit):
        config_from_args([])
    assert GraphRouteConfig(dataset="test").data_dir == "data"


def test_flags_override_the_script_without_discarding_its_other_values():
    base = GraphRouteConfig(dataset="test", graph={"k": 3},
                            gnn={"arch": "gat", "epochs": 40})
    cfg = config_from_args(["--graph-k", "11"], base=base)
    assert cfg.graph.k == 11
    assert cfg.gnn.arch == "gat"
    assert cfg.gnn.epochs == 40


def test_boolean_and_list_flags_are_generated_from_field_types():
    cfg = config_from_args([
        "--dataset", "test",
        "--graph-pool-calibrate", "false",
        "--base-models", "mlp32", "mlp64",
    ])
    assert cfg.graph.pool_calibrate is False
    assert cfg.base.models == ["mlp32", "mlp64"]


def test_gnn_namespace_covers_what_train_gnn_reads():
    """train_gnn and evaluate_test read a flat args object, and federated callers
    build their own. If this drifts, those callers break silently."""
    import re
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = open(os.path.join(root, "graphroute", "training.py")).read()
    bodies = src[src.index("def train_gnn("):]
    # both spellings: args.x, and getattr(args, "x", default) -- the second is
    # how optional fields are read, and is exactly where a missing one hides,
    # because getattr's default makes the omission silent instead of loud.
    needed = set(re.findall(r"args\.([a-z_0-9]+)", bodies))
    needed |= set(re.findall(r'getattr\(\s*args\s*,\s*"([a-z_0-9]+)"', bodies))
    have = set(vars(GraphRouteConfig(dataset="test").gnn_namespace()))
    assert needed <= have, (
        f"training reads settings not in gnn_namespace(): {needed - have}")


def test_voting_space_follows_the_loss_target():
    """Ensemble training weights by sigmoid, so inference should too."""
    assert GraphRouteConfig(dataset="test", loss_target="ensemble").resolved_voting_weight_space() == "sig"
    assert GraphRouteConfig(dataset="test", loss_target="meta_labels").resolved_voting_weight_space() == "logit"
    explicit = GraphRouteConfig(dataset="test", loss_target="ensemble",
                                gnn={"voting_weight_space": "logit"})
    assert explicit.resolved_voting_weight_space() == "logit"   # explicit wins


def test_invalid_values_are_refused():
    with pytest.raises(Exception):
        GNNConfig(arch="not_an_arch")
    with pytest.raises(Exception):
        GNNConfig(hidden_dim=0)
    with pytest.raises(Exception):
        GNNConfig(unknown_field=1)          # extra="forbid" catches typos


def test_pair_feat_dim_matches_what_the_head_will_receive():
    """The head's first Linear is sized from this, so a wrong width is a shape
    error at the first forward -- it was hardcoded to 0 regardless of config."""
    cfg = GraphRouteConfig(dataset="test", num_classes=5,
                           gnn={"output_head": "concat_mlp",
                                               "pair_confidence": True,
                                               "pair_competence": "gain"})
    assert cfg.pair_feat_dim() == 6                    # 5 class probabilities + 1 gain
    assert GraphRouteConfig(dataset="test", num_classes=5).pair_feat_dim() == 0


def test_pair_only_without_pair_features_is_refused():
    with pytest.raises(ValueError, match="nothing to read"):
        GraphRouteConfig(dataset="test", gnn={
            "output_head": "concat_mlp", "pair_only": True}).validate_head()


def test_every_architecture_can_be_constructed():
    """gnn_kwargs sent every setting to every architecture, so two of the three
    raised TypeError on unexpected keyword arguments."""
    from graphroute.gnn import build_gnn
    for arch in ("gat", "graph_gps", "mlp"):
        cfg = GraphRouteConfig(dataset="test", gnn={"arch": arch})
        build_gnn(arch, **cfg.gnn_kwargs(input_dim=12, out_dim=3))


def test_regression_resolves_classification_only_defaults():
    cfg = GraphRouteConfig(dataset="test", task="regression")
    assert cfg.num_classes == 1
    assert cfg.graph.pool_calibrate is False
    assert cfg.base.weighted_by_class is False
    assert cfg.base.es_metric == cfg.gnn.es_metric == "val_loss"
    assert cfg.gnn.ens_combination_mode == "weighted_mean"


@pytest.mark.parametrize("overrides,match", [
    ({"num_classes": 2}, "one model output"),
    ({"graph": {"pool_calibrate": True}}, "class probabilities"),
    ({"graph": {"neighbor_mode": "class_balanced"}}, "class_balanced"),
    ({"graph": {"weight_mode": "cmdw"}}, "class membership"),
    ({"base": {"es_metric": "val_acc"}}, "base models"),
    ({"gnn": {"es_metric": "val_bacc"}}, "GNN training"),
    ({"gnn": {"fallback": "acc"}}, "fallback"),
    ({"gnn": {"pair_confidence": True}}, "class probabilities"),
    ({"gnn": {"sample_weight_mode": "difficulty"}}, "sample_weight_mode"),
    ({"gnn": {"ens_combination_mode": "hard_voting"}}, "weighted mean"),
    ({"gnn": {"loss": "soft_bce"}}, "classification margins"),
])
def test_regression_refuses_classification_only_settings(overrides, match):
    with pytest.raises(Exception, match=match):
        GraphRouteConfig(dataset="test", task="regression", **overrides)


def test_classification_still_accepts_classification_settings():
    GraphRouteConfig(dataset="test", task="classification",
                     graph={"neighbor_mode": "class_balanced"},
                     gnn={"es_metric": "val_bacc"})          # fine for classification
    with pytest.raises(Exception, match="for regression"):
        GraphRouteConfig(dataset="test",
                         gnn={"ens_combination_mode": "weighted_mean"})
