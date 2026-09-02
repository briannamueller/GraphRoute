"""Reusable pool weights, OOF outputs, and raw inference outputs."""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.data._utils.collate import default_collate

from graphroute.pool_cache import (cached_pool, fingerprint_model, load_pool,
                                   save_pool)


class _ScaledLinear(nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = scale
        self.linear = nn.Linear(4, 2)

    def forward(self, x):
        return self.linear(x) * self.scale


class _Box:
    def __init__(self, value):
        self.value = value


class _BoxDataset(Dataset):
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return _Box(self.x[index]), self.y[index]


def _box_collate(batch):
    return (torch.stack([box.value for box, _ in batch]),
            torch.stack([target for _, target in batch]))


def _factories(n=2, out=3):
    return [lambda: nn.Linear(4, out) for _ in range(n)]


def _trained(factories):
    return [f() for f in factories]


CPU = torch.device("cpu")
IDS = ["linear-a", "linear-b"]
FP = "pool-fingerprint"


def test_first_request_trains_and_the_second_loads(tmp_path):
    factories = _factories()
    calls = {"n": 0}

    def train():
        calls["n"] += 1
        return _trained(factories), None

    first = cached_pool(tmp_path / "pool", factories, train,
                        fingerprint_value=FP, model_ids=IDS)
    second = cached_pool(tmp_path / "pool", factories, train,
                         fingerprint_value=FP, model_ids=IDS)

    assert calls["n"] == 1
    assert second.models is None
    for a, b in zip(first.load_models(CPU), second.load_models(CPU)):
        assert torch.allclose(a.weight, b.weight)          # the same pool, restored


def test_an_empty_directory_is_a_miss(tmp_path):
    assert load_pool(tmp_path / "nothing", _factories(),
                     fingerprint_value=FP, model_ids=IDS) is None


def test_a_partial_pool_is_a_miss(tmp_path):
    """Half a pool is not a pool."""
    factories = _factories()
    save_pool(tmp_path / "pool", _trained(factories),
              fingerprint_value=FP, model_ids=IDS)
    (tmp_path / "pool" / "models" / "model_1.pt").unlink()
    assert load_pool(tmp_path / "pool", factories,
                     fingerprint_value=FP, model_ids=IDS) is None


def test_oof_logits_round_trip(tmp_path):
    factories = _factories()
    models = _trained(factories)
    oof = torch.randn(20, len(models), 3)

    save_pool(tmp_path / "pool", models, fingerprint_value=FP,
              model_ids=IDS, oof_logits=oof)
    loaded = load_pool(tmp_path / "pool", factories,
                       fingerprint_value=FP, model_ids=IDS, require_oof=True)

    assert loaded is not None
    assert torch.allclose(loaded.load_oof(), oof)


def test_a_pool_without_its_oof_logits_cannot_serve_an_oof_run(tmp_path):
    """Silently serving it would answer with in-sample predictions.

    The stored models were retrained on all of the training data, so asking them
    about those rows again is exactly what oof_stacking exists to avoid.
    """
    factories = _factories()
    save_pool(tmp_path / "pool", _trained(factories),
              fingerprint_value=FP, model_ids=IDS)       # models only

    assert load_pool(tmp_path / "pool", factories, fingerprint_value=FP,
                     model_ids=IDS, require_oof=True) is None
    assert load_pool(tmp_path / "pool", factories, fingerprint_value=FP,
                     model_ids=IDS, require_oof=False) is not None


def test_the_driver_derives_and_reuses_the_pool(tmp_path, monkeypatch):
    import graphroute.run as run_mod
    from graphroute.config import GraphRouteConfig

    calls = {"n": 0}
    real = run_mod.train_pool_oof

    def counted(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(run_mod, "train_pool_oof", counted)
    monkeypatch.chdir(tmp_path)

    torch.manual_seed(0)
    x = torch.randn(24, 4)
    y = (x[:, 0] > 0).long()
    data = TensorDataset(x, y)
    cfg = GraphRouteConfig(dataset="driver-data", num_classes=2, seed=0,
                           graph={"pool_calibrate": False, "k": 2},
                           gnn={"arch": "mlp", "epochs": 1, "patience": 1})
    cfg.base.oof_folds = 2
    cfg.base.epochs = 1
    cfg.base.batch_size = 8
    template = nn.Linear(4, 2)

    kw = dict(models=[template])
    first = run_mod.fit_graphroute(cfg, data, validation_set=data, **kw)
    second = run_mod.fit_graphroute(cfg, data, validation_set=data, **kw)
    assert calls["n"] == 1
    assert first.pool.directory.resolve() == (
        tmp_path / "pool_cache" / "driver-data" /
        f"pool_{first.pool.fingerprint}").resolve()
    import json
    manifest = json.loads((first.pool.directory / "manifest.json").read_text())
    assert manifest["data_id"] == "driver-data"
    for name, value in first.gnn.state_dict().items():
        assert torch.equal(value, second.gnn.state_dict()[name])


def test_model_templates_are_untouched_and_cache_without_manual_ids(
        tmp_path, monkeypatch):
    import graphroute.run as run_mod
    from graphroute.config import GraphRouteConfig

    calls = {"n": 0}
    real = run_mod.train_pool_oof

    def counted(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(run_mod, "train_pool_oof", counted)
    x = torch.randn(24, 4, generator=torch.Generator().manual_seed(2))
    data = TensorDataset(x, (x[:, 0] > 0).long())
    template = nn.Linear(4, 2)
    initial = {name: value.clone() for name, value in template.state_dict().items()}
    cfg = GraphRouteConfig(
        dataset="template-data", num_classes=2, seed=0,
        base={"oof_folds": 2, "epochs": 1, "batch_size": 8},
        graph={"pool_calibrate": False, "k": 2},
        gnn={"arch": "mlp", "epochs": 1, "patience": 1})

    first = run_mod.fit_graphroute(
        cfg, data, validation_set=data, models=[template], cache_dir=tmp_path)
    second = run_mod.fit_graphroute(
        cfg, data, validation_set=data, models=[template], cache_dir=tmp_path)

    assert calls["n"] == 1
    assert first.pool.model_ids == second.pool.model_ids
    assert all(torch.equal(value, initial[name])
               for name, value in template.state_dict().items())


def test_dataset_name_is_the_readable_cache_identity(tmp_path):
    from graphroute.pool_cache import pool_directory

    assert pool_directory(tmp_path, "cifar10-split-v1", FP) == (
        tmp_path / "cifar10-split-v1" / f"pool_{FP}")


def test_model_fingerprint_tracks_non_tensor_behavior_settings():
    a = _ScaledLinear(1.0)
    b = _ScaledLinear(2.0)
    b.load_state_dict(a.state_dict())

    assert not torch.equal(a(torch.ones(2, 4)), b(torch.ones(2, 4)))
    assert fingerprint_model(a) == fingerprint_model(copy.deepcopy(a))
    assert fingerprint_model(a) != fingerprint_model(b)


def test_only_an_active_derived_validation_ratio_changes_pool_identity(
        tmp_path, monkeypatch):
    import graphroute.run as run_mod
    from graphroute.config import GraphRouteConfig

    seen = []

    def capture(*args, fingerprint_value, **kwargs):
        seen.append(fingerprint_value)
        raise RuntimeError("captured")

    monkeypatch.setattr(run_mod, "cached_pool", capture)
    data = TensorDataset(torch.randn(20, 4), torch.arange(20) % 2)
    template = nn.Linear(4, 2)

    def identity(ratio, validation_set=None):
        cfg = GraphRouteConfig(
            dataset="ratio-data", num_classes=2, val_ratio=ratio,
            base={"epochs": 1, "oof_folds": 2},
            graph={"pool_calibrate": False},
            gnn={"arch": "mlp", "epochs": 1})
        with pytest.raises(RuntimeError, match="captured"):
            run_mod.fit_graphroute(
                cfg, data, validation_set=validation_set,
                models=[template], cache_dir=tmp_path)
        return seen[-1]

    assert identity(0.2) != identity(0.4)
    assert identity(0.2, data) == identity(0.4, data)


def test_persistent_custom_collation_uses_the_dataset_name(tmp_path):
    from graphroute.config import GraphRouteConfig
    from graphroute.run import fit_graphroute

    x = torch.randn(24, 4)
    data = TensorDataset(x, (x[:, 0] > 0).long())
    cfg = GraphRouteConfig(
        dataset="doubled-data", num_classes=2, seed=0,
        base={"split_mode": "split_train", "epochs": 1, "batch_size": 8},
        graph={"pool_calibrate": False, "k": 2},
        gnn={"arch": "mlp", "epochs": 1, "patience": 1})

    def doubled(batch):
        values, targets = default_collate(batch)
        return values * 2, targets

    fitted = fit_graphroute(
        cfg, data, validation_set=data, models=[nn.Linear(4, 2)],
        cache_dir=tmp_path, collate_fn=doubled)
    assert fitted.pool.directory.parent.name == "doubled-data"


def test_named_dataset_handles_structured_samples_without_content_hashing(tmp_path):
    from graphroute.config import GraphRouteConfig
    from graphroute.run import fit_graphroute

    generator = torch.Generator().manual_seed(4)
    x = torch.randn(24, 4, generator=generator)
    train = _BoxDataset(x, (x[:, 0] > 0).long())
    later_x = torch.randn(12, 4, generator=generator)
    later = _BoxDataset(later_x, (later_x[:, 0] > 0).long())
    cfg = GraphRouteConfig(
        dataset="boxed-data", num_classes=2, seed=0,
        base={"split_mode": "split_train", "epochs": 1, "batch_size": 8},
        graph={"pool_calibrate": False, "k": 2},
        gnn={"arch": "mlp", "epochs": 1, "patience": 1})

    fitted = fit_graphroute(
        cfg, train, validation_set=train, models=[nn.Linear(4, 2)],
        cache_dir=tmp_path, collate_fn=_box_collate)
    before = set(fitted.pool.output_directory.glob("*.pt"))
    metrics = fitted.evaluate(later)
    after = set(fitted.pool.output_directory.glob("*.pt"))

    assert "accuracy" in metrics
    assert after - before == {fitted.pool.output_directory / "test_logits.pt"}
    assert {path.name for path in before} == {
        "train_logits.pt", "validation_logits.pt"
    }


def test_cached_outputs_do_not_load_model_weights(tmp_path):
    factories = _factories()
    artifact = save_pool(tmp_path / "pool", _trained(factories),
                         fingerprint_value=FP, model_ids=IDS)
    data = TensorDataset(torch.randn(8, 4), torch.zeros(8, dtype=torch.long))
    loader = DataLoader(data, batch_size=4)
    expected = artifact.cached_outputs("validation_logits", loader, CPU,
                                       task="classification")
    restored = load_pool(tmp_path / "pool", factories,
                         fingerprint_value=FP, model_ids=IDS)
    restored.model_factories = tuple(lambda: (_ for _ in ()).throw(
        AssertionError("weights were loaded")) for _ in factories)
    actual = restored.cached_outputs("validation_logits", loader, CPU,
                                     task="classification")
    assert torch.equal(actual, expected)


def test_output_transform_is_applied_before_publication(tmp_path):
    factories = _factories()
    artifact = save_pool(tmp_path / "pool", _trained(factories),
                         fingerprint_value=FP, model_ids=IDS)
    data = TensorDataset(torch.randn(8, 4), torch.zeros(8, dtype=torch.long))
    loader = DataLoader(data, batch_size=4)
    value = artifact.cached_outputs(
        "train_logits", loader, CPU, task="classification",
        transform=lambda logits: torch.zeros_like(logits))
    assert torch.count_nonzero(value) == 0
    stored = torch.load(tmp_path / "pool" / "outputs" / "train_logits.pt",
                        weights_only=True)
    assert torch.count_nonzero(stored) == 0


def test_repairing_an_incomplete_cache_is_never_half_visible(tmp_path, monkeypatch):
    """A reader must not get old models carrying the new fold logits.

    The window: a directory holds models from an earlier pool but no logits, so
    an oof_stacking run misses, retrains and republishes. ``save_pool`` writes
    the logits first, then the models one by one -- and between those two steps
    every file a reader looks for exists, while half of them belong to the pool
    that was just discarded. The writer is paused exactly there.
    """
    import threading
    import time

    from graphroute import pool_cache
    from graphroute.pool_cache import OOF_NAME, _atomic_save

    pool = tmp_path / "pool"
    factories = _factories()

    stale = _trained(factories)
    for model in stale:                                   # recognisably the old pool
        torch.nn.init.constant_(model.weight, 1.0)
    save_pool(pool, stale, fingerprint_value=FP, model_ids=IDS)

    fresh = _trained(factories)
    for model in fresh:
        torch.nn.init.constant_(model.weight, 2.0)
    fresh_oof = torch.full((8, len(fresh), 3), 7.0)

    logits_published = threading.Event()
    may_finish = threading.Event()
    real_save = pool_cache.save_pool

    def paused_save(directory, models, *, fingerprint_value, model_ids,
                    oof_logits=None, data_id=None):
        _atomic_save(oof_logits, pool / OOF_NAME)
        logits_published.set()
        assert may_finish.wait(timeout=10)
        return real_save(directory, models, fingerprint_value=fingerprint_value,
                         model_ids=model_ids, oof_logits=oof_logits,
                         data_id=data_id)

    monkeypatch.setattr(pool_cache, "save_pool", paused_save)

    def writer():
        pool_cache.cached_pool(
            pool, factories, lambda: (fresh, fresh_oof),
            fingerprint_value=FP, model_ids=IDS, require_oof=True)

    reader_trained = {"n": 0}
    seen = {}

    def reader():
        def train():                                      # must not be reached
            reader_trained["n"] += 1
            return fresh, fresh_oof
        seen["pool"] = pool_cache.cached_pool(
            pool, factories, train, fingerprint_value=FP,
            model_ids=IDS, require_oof=True)

    w = threading.Thread(target=writer)
    w.start()
    assert logits_published.wait(timeout=10)              # writer is inside the window

    r = threading.Thread(target=reader)
    r.start()
    time.sleep(0.2)                                       # long enough to read, if it could
    assert not seen, "the reader loaded the directory mid-publish"

    may_finish.set()
    w.join(timeout=10)
    r.join(timeout=10)
    assert not w.is_alive() and not r.is_alive()

    artifact = seen["pool"]
    assert reader_trained["n"] == 0                       # it waited, then hit the cache
    for model in artifact.load_models(CPU):
        assert torch.allclose(model.weight, torch.full_like(model.weight, 2.0)), \
            "reader got models from the discarded pool"
    assert torch.allclose(artifact.load_oof(), fresh_oof)
