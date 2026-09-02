"""Property test: an OOF prediction must come from a model that never saw that row.

Wraps fit_classifier to record exactly which dataset indices each fold's model
was trained and early-stopped on, then asserts those are disjoint from the
indices whose predictions that fold contributes.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

import graphroute.pool as pool
from graphroute.pool import train_pool_oof

N_CLASSES, N_FEATURES = 3, 10
seen = []          # (train_idx, val_idx) per fit_classifier call

_real_fit = pool.fit_classifier


def _recording_fit(model, train_loader, val_loader, device, **kw):
    def idx(loader):
        d = loader.dataset
        return set(np.asarray(d.indices).tolist()) if hasattr(d, "indices") else None
    seen.append((idx(train_loader), idx(val_loader),
                 id(train_loader.dataset), id(val_loader.dataset)))
    return _real_fit(model, train_loader, val_loader, device, **kw)




def blobs(n_per_class, gen, centers):
    xs = [centers[c] + torch.randn(n_per_class, N_FEATURES, generator=gen) * 2.0
          for c in range(N_CLASSES)]
    ys = [torch.full((n_per_class,), c, dtype=torch.long) for c in range(N_CLASSES)]
    return TensorDataset(torch.cat(xs), torch.cat(ys))


def test_oof_predictions_are_out_of_sample(monkeypatch):
    monkeypatch.setattr(pool, "fit_classifier", _recording_fit)
    seen.clear()
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    centers = torch.randn(N_CLASSES, N_FEATURES, generator=gen) * 1.5
    train_set, val_set = blobs(40, gen, centers), blobs(20, gen, centers)

    factories = [lambda: nn.Sequential(nn.Linear(N_FEATURES, 16), nn.ReLU(),
                                       nn.Linear(16, N_CLASSES))] * 2
    n_folds = 3

    models, oof_logits, labels = train_pool_oof(
        factories, train_set, val_set, torch.device("cpu"),
        n_folds=n_folds, batch_size=16, max_epochs=15, patience=5,
        num_classes=N_CLASSES, seed=0,
    )

    N, M = len(train_set), len(factories)
    print("\n--- assertions ---")

    assert oof_logits.shape == (N, M, N_CLASSES), oof_logits.shape
    print(f"  oof_logits shape {tuple(oof_logits.shape)}                     OK")

    unpopulated = int((oof_logits.abs().sum(dim=(1, 2)) == 0).sum())
    assert unpopulated == 0, f"{unpopulated} rows never predicted"
    print("  every row predicted (0 unpopulated)                OK")

    # Recompute the outer folds the function used, then check disjointness.
    from sklearn.model_selection import StratifiedKFold
    y = np.array([int(train_set[i][1]) for i in range(N)])
    folds = list(StratifiedKFold(n_splits=n_folds, shuffle=True,
                                 random_state=0).split(np.arange(N), y))

    fold_calls = seen[:n_folds * M]
    for f, (_, oof_idx) in enumerate(folds):
        held = set(oof_idx.tolist())
        for m in range(M):
            tr, va, _, _ = fold_calls[f * M + m]
            assert tr is not None and va is not None
            assert not (tr & held), f"fold {f} model {m}: trained on {len(tr & held)} held-out rows"
            assert not (va & held), f"fold {f} model {m}: early-stopped on {len(va & held)} held-out rows"
            assert not (tr & va), f"fold {f} model {m}: inner train/val overlap"
    print("  held-out fold unseen in training AND early stop    OK")
    print("  inner train/val disjoint                           OK")

    tr0, va0, _, _ = fold_calls[0]
    print(f"\n  fold 0: inner-train {len(tr0)}, inner-val {len(va0)}, "
          f"held-out {len(folds[0][1])}  (total {N})")

    retrain = seen[n_folds * M:]
    assert len(retrain) == M
    for tr, va, tr_id, va_id in retrain:
        assert tr is None, "retrain train loader should wrap the full dataset, not a Subset"
        assert va is None, "retrain val should be the separate val_dataset, not a Subset"
        assert tr_id == id(train_set), "retrain must train on the full training set"
        assert va_id == id(val_set), "retrain must validate on the held-out val_dataset"
    print("  retrain trains on train_set, selects on val_set    OK")
    fold_ds_ids = {i for _, _, a, b in fold_calls for i in (a, b)}
    assert id(val_set) not in fold_ds_ids, "val_dataset must not be touched during folds"
    print("  val_dataset untouched during the fold stage        OK")
    print("\nall assertions passed")


if __name__ == "__main__":                      # still runnable directly
    pool.fit_classifier = _recording_fit
    test_oof_predictions_are_out_of_sample(type("_", (), {"setattr": staticmethod(setattr)})())
