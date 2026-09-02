"""GraphRoute end-to-end on synthetic data.

Trains a small heterogeneous pool, builds the decision-space graph, trains the
GNN meta-learner, and compares per-sample selection against the pool. Runs on
CPU in seconds and downloads nothing. It demonstrates the library workflow,
not a benchmark result.

    python quickstart.py

On a busy shared login node, set OMP_NUM_THREADS=1 -- torch's default thread
pool thrashes there and turns a 15-second run into several minutes.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from graphroute.config import GraphRouteConfig
from graphroute.run import fit_graphroute

SEED, N_CLASSES, N_FEATURES, SUBSPACE = 0, 4, 24, 8


class SubspaceMLP(nn.Module):
    """An MLP that only sees SUBSPACE of the features.

    Each pool member gets a different subset, so no classifier dominates: each is
    competent on the samples its own features happen to separate. That
    complementarity is what dynamic selection exploits -- with an
    all-see-everything pool there is nothing to select between.
    """

    def __init__(self, feature_idx, width):
        super().__init__()
        self.register_buffer("idx", feature_idx)
        self.net = nn.Sequential(
            nn.Linear(len(feature_idx), width), nn.ReLU(), nn.Linear(width, N_CLASSES))

    def forward(self, x):
        return self.net(x[:, self.idx])


def make_blobs(n_per_class, centers, generator):
    """Gaussian blobs around shared centers: separable, but noisy enough that no
    single classifier wins on every sample."""
    xs, ys = [], []
    for c in range(N_CLASSES):
        xs.append(centers[c] + torch.randn(n_per_class, N_FEATURES, generator=generator) * 2.4)
        ys.append(torch.full((n_per_class,), c, dtype=torch.long))
    x, y = torch.cat(xs), torch.cat(ys)
    perm = torch.randperm(len(y), generator=generator)
    return TensorDataset(x[perm], y[perm])


def main():
    torch.manual_seed(SEED)
    gen = torch.Generator().manual_seed(SEED)

    # One set of class centers shared by all splits -- otherwise the splits
    # describe unrelated problems and every metric collapses to chance.
    centers = torch.randn(N_CLASSES, N_FEATURES, generator=gen) * 1.1
    train_set, test_set = (make_blobs(n, centers, gen) for n in (300, 250))

    subspaces = [torch.randperm(N_FEATURES, generator=gen)[:SUBSPACE] for _ in range(4)]
    models = [SubspaceMLP(idx, width)
              for idx, width in zip(subspaces, (16, 32, 64, 32))]

    cfg = GraphRouteConfig(
        dataset="synthetic-blobs", num_classes=N_CLASSES, device="cpu", seed=SEED,
        base={"epochs": 60, "es_patience": 10, "lr": 1e-3, "batch_size": 32},
        graph={"k": 5, "pool_calibrate": False},
        gnn={"arch": "gat", "hidden_dim": 64, "epochs": 200, "patience": 25,
             "es_metric": "val_acc",
             "ens_combination_mode": "hard_weighted_voting", "voting_weight_space": "sig"},
    )
    model = fit_graphroute(cfg, train_set, models=models)
    test_metrics = model.evaluate(test_set)

    # Compare against the pool it selects from, at matched combination rules:
    # selection is the only thing that differs within each pair.
    pool = model.pool.load_models(torch.device("cpu"))
    from graphroute.pool import collect_pool_predictions
    loader = DataLoader(test_set, batch_size=256)
    y = torch.cat([b for _, b in loader])
    probs = torch.softmax(collect_pool_predictions(pool, loader, torch.device("cpu")), -1)
    hard = probs.argmax(-1)
    acc = lambda p: (p == y).float().mean().item()

    print("\n--- test accuracy ---")
    for i in range(len(pool)):
        print(f"  classifier {i}                    {acc(hard[:, i]):.4f}")
    print(f"  uniform ensemble (soft)         {acc(probs.mean(1).argmax(-1)):.4f}")
    print(f"  GraphRoute                      {test_metrics['accuracy']:.4f}")
    onehot = torch.nn.functional.one_hot(hard, N_CLASSES).float()
    print(f"  uniform ensemble (hard vote)    {acc(onehot.mean(1).argmax(-1)):.4f}")
    oracle = torch.where((hard == y.unsqueeze(1)).any(1), y, hard[:, 0])
    print(f"  oracle (any classifier right)   {acc(oracle):.4f}   <- selection headroom")


if __name__ == "__main__":
    main()
