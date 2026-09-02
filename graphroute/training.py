"""GNN training, evaluation, and classification fallback behavior."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData

from graphroute.losses import (
    compute_ensemble_loss,
    compute_meta_loss,
    margin_targets,
)
from graphroute.selection import (
    compute_selection_matrix,
    evaluate_ensemble,
    evaluate_ensemble_regression,
)


# ── Sample weights ─────────────────────────────────────────────────────

def compute_sample_weights(
    mode: str,
    labels: torch.Tensor,
    ds: torch.Tensor,
    num_classes: int,
    task: str = "classification",
) -> torch.Tensor | None:
    """Compute per-sample weights for GNN training.

    Args:
        mode: "none", "class_prevalence", or "difficulty".
        labels: Ground truth labels [N].
        ds: Decision-space tensor [N, M*C].
        num_classes: Number of classes.
        task: "classification" or "regression".

    Returns:
        Weights tensor [N] (mean-normalized) or None.
    """
    if mode == "none":
        return None

    N = labels.size(0)

    if mode == "class_prevalence" and task == "classification":
        counts = torch.zeros(num_classes, device=labels.device)
        for c in range(num_classes):
            counts[c] = (labels == c).sum().float()
        counts = counts.clamp(min=1.0)
        freq = counts / counts.sum()
        w = 1.0 / (freq[labels] * num_classes)
        w = w / w.mean()
        return w

    if mode == "difficulty":
        M = ds.size(1) // max(num_classes, 1)
        probs = ds.view(N, M, num_classes).float()
        if task == "classification":
            y_idx = labels.view(-1, 1, 1).expand(-1, M, 1)
            p_true = probs.gather(2, y_idx).squeeze(-1)  # [N, M]
            avg_conf = p_true.mean(dim=1)  # [N]
            w = 1.0 - avg_conf
        else:
            w = torch.ones(N, device=labels.device)
        w = w.clamp(min=0.01)
        w = w / w.mean()
        return w

    return None


# ── Pair features ──────────────────────────────────────────────────────

def compute_pair_features(
    ds: torch.Tensor,
    num_classes: int,
    pair_confidence: bool = False,
    pair_competence: str = "none",
    data: HeteroData | None = None,
    meta: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Compute per-(sample, pool-member) features for the concat_mlp head.

    Args:
        ds: Decision-space tensor [N, M*C].
        num_classes: Number of classes.
        pair_confidence: Include classifier probability vector as feature.
        pair_competence: "none" or "gain" (neighborhood gain feature).
        data: HeteroData graph (needed for gain computation).
        meta: Meta-labels [N, M] (needed for gain computation).

    Returns:
        Pair features [N, M, P] or None.
    """
    if not pair_confidence and pair_competence == "none":
        return None

    M = ds.size(1) // max(num_classes, 1)
    parts = []

    if pair_confidence:
        probs = ds.view(-1, M, num_classes)  # [N, M, C]
        parts.append(probs)

    if pair_competence == "gain" and data is not None and meta is not None:
        gain = _neighborhood_gain(data, meta)  # [N, M]
        parts.append(gain.unsqueeze(-1))  # [N, M, 1]

    if not parts:
        return None
    return torch.cat(parts, dim=-1)


def _neighborhood_gain(data: HeteroData, meta: torch.Tensor) -> torch.Tensor:
    """Compute per-(sample, pool-member) neighborhood competence gain.

    G(m, i) = neigh_acc_m(i) - mean_j(neigh_acc_m(j))
    """
    rel = ("sample", "ss", "sample")
    ei = data[rel].edge_index
    eattr = getattr(data[rel], "edge_attr", None)

    src, dst = ei[0], ei[1]
    N, M = meta.shape

    if eattr is not None:
        w = eattr.float().view(-1)
    else:
        w = torch.ones(src.numel(), device=src.device)

    # Weighted sum of neighbor competence. Allocate through meta so these follow
    # the graph onto whatever device it is on rather than defaulting to CPU.
    weighted_correct = w.unsqueeze(1) * meta[src]  # [E, M]
    score_num = meta.new_zeros(N, M)
    score_den = meta.new_zeros(N, 1)
    score_num.scatter_add_(0, dst.unsqueeze(1).expand_as(weighted_correct), weighted_correct)
    score_den.scatter_add_(0, dst.unsqueeze(1), w.unsqueeze(1))
    score_den = score_den.clamp(min=1e-8)
    neigh_acc = score_num / score_den  # [N, M]

    mean_acc = neigh_acc.mean(dim=0, keepdim=True)  # [1, M]
    return neigh_acc - mean_acc


# ── Fallback model ─────────────────────────────────────────────────────

class FallbackModel:
    """Classification fallback based on neighborhood accuracy.

    Used for samples where the GNN selects nobody. Each mode scores classifier m
    for sample i from how well m did on i's training neighbours:

    ``acc``   unweighted mean correctness over the neighbourhood
    ``wacc``  the same, weighted by edge weight (closer neighbours count more)
    ``bacc``  mean per-class recall over the neighbourhood, so a classifier that
              only handles the common class there does not look competent

    """

    def __init__(self, mode: str, meta: torch.Tensor, labels: torch.Tensor):
        self.mode = mode
        self.meta = meta.float()
        self.labels = labels.long()

    def __call__(self, data: HeteroData, eval_mask: torch.Tensor) -> torch.Tensor:
        train_mask = data["sample"].train_mask.bool()
        # Built from CPU tensors at construction; the graph may be elsewhere.
        device = data["sample"].x.device
        self.meta = self.meta.to(device)
        self.labels = self.labels.to(device)
        N = data["sample"].x.size(0)
        M = self.meta.size(1)

        rel = ("sample", "ss", "sample")
        ei = data[rel].edge_index
        eattr = getattr(data[rel], "edge_attr", None)

        src, dst = ei[0], ei[1]
        keep = train_mask[src]
        src, dst = src[keep], dst[keep]

        if self.mode == "wacc" and eattr is not None:
            w = eattr.float().view(-1)[keep]
        else:
            w = torch.ones(src.numel(), device=src.device)

        # Map global train indices to local meta-label rows, on the graph's device
        train_nodes = train_mask.nonzero(as_tuple=False).view(-1)
        g2l = torch.full((N,), -1, dtype=torch.long, device=src.device)
        g2l[train_nodes] = torch.arange(train_nodes.numel(), device=src.device)
        src_local = g2l[src]

        if self.mode == "bacc":
            return self._balanced(N, M, dst, src_local)[eval_mask]

        weighted_correct = w.unsqueeze(1) * self.meta[src_local]
        score_num = self.meta.new_zeros(N, M)
        score_den = self.meta.new_zeros(N, 1)
        score_num.scatter_add_(0, dst.unsqueeze(1).expand_as(weighted_correct), weighted_correct)
        score_den.scatter_add_(0, dst.unsqueeze(1), w.unsqueeze(1))
        score_den = score_den.clamp(min=1e-8)
        scores = score_num / score_den

        return scores[eval_mask]

    def _balanced(self, N: int, M: int, dst: torch.Tensor,
                  src_local: torch.Tensor) -> torch.Tensor:
        """Mean per-class recall over each node's neighbourhood.

        Accumulate correctness per (node, true class) so each class contributes
        its own recall, then average over the classes actually present -- an
        unweighted mean would let the neighbourhood's majority class decide.
        """
        C = int(self.labels.max().item()) + 1
        src_class = self.labels[src_local]
        key = dst * C + src_class                       # one bucket per (node, class)

        num = self.meta.new_zeros(N * C, M)
        den = self.meta.new_zeros(N * C, 1)
        num.scatter_add_(0, key.unsqueeze(1).expand(-1, M), self.meta[src_local])
        den.scatter_add_(0, key.unsqueeze(1), den.new_ones(key.numel(), 1))

        present = (den.view(N, C, 1) > 0).float()       # classes seen at each node
        recall = (num.view(N, C, M) / den.view(N, C, 1).clamp(min=1e-8)) * present
        return recall.sum(dim=1) / present.sum(dim=1).clamp(min=1e-8)


# ── GNN Training Loop ─────────────────────────────────────────────────

def train_gnn(
    gnn: torch.nn.Module,
    data: HeteroData,
    meta: dict,
    args,
    device: torch.device,
) -> tuple[torch.nn.Module, dict]:
    """Train the GNN meta-learner with early stopping.

    Args:
        gnn: GNN model.
        data: HeteroData graph (train + val).
        meta: Metadata dict from build_graph.
        args: Parsed arguments namespace.
        device: Compute device.

    Returns:
        (trained_gnn, history dict with best metrics).
    """
    if args.gnn_task == "regression" and args.gnn_es_metric != "val_loss":
        raise ValueError("Regression GNN training supports val_loss early stopping.")
    if (args.gnn_task == "regression"
            and args.gnn_loss_target == "meta_labels"
            and args.gnn_loss == "soft_bce"):
        raise ValueError(
            "soft_bce uses classification margins; choose bce or regression for "
            "regression competence targets.")
    gnn = gnn.to(device)
    data = data.to(device)

    combined_ds = meta["combined_ds"].to(device)
    combined_meta = meta["combined_meta"].to(device)
    combined_labels = meta["combined_labels"].to(device)
    train_mask = data["sample"].train_mask.bool()
    val_mask = data["sample"].val_mask.bool()
    n_train = meta["n_train"]
    num_classes = args.num_classes

    # Precompute margin targets if needed (classification only — margins are
    # undefined for regression since there are no class probabilities)
    train_margin = None
    if (args.gnn_loss_target == "meta_labels"
            and args.gnn_loss in ("soft_bce", "regression")
            and args.gnn_task == "classification"):
        train_margin = margin_targets(combined_ds, combined_labels, num_classes).to(device)

    # Compute sample weights
    sample_weights = compute_sample_weights(
        args.gnn_sample_weight_mode,
        combined_labels[:n_train],
        combined_ds[:n_train],
        num_classes,
        task=args.gnn_task,
    )
    if sample_weights is not None:
        sample_weights = sample_weights.to(device)

    # Compute pair features
    pair_feats = None
    if args.gnn_output_head in ("concat_mlp",):
        pair_feats = compute_pair_features(
            combined_ds, num_classes,
            pair_confidence=args.gnn_output_head_pair_confidence,
            pair_competence=args.gnn_output_head_pair_competence,
            data=data, meta=combined_meta,
        )
        if pair_feats is not None:
            data["sample"].pair_feats = pair_feats.to(device)

    optimizer = torch.optim.Adam(gnn.parameters(), lr=args.gnn_lr,
                                 weight_decay=args.gnn_weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=max(1, args.gnn_patience // 3), factor=0.5,
    )

    best_state = None
    best_metric = -float("inf")
    best_epoch = 0
    patience_counter = 0
    history = []

    train_nodes = torch.where(train_mask)[0]
    batch_size = getattr(args, "gnn_batch_size", 0) or 0

    for epoch in range(1, args.gnn_epochs + 1):
        # ── Train ──
        gnn.train()
        logits = gnn(data)

        # Message passing stays on the full training-context graph: a training
        # node can depend on training neighbours not selected for this loss
        # batch. Batching subsets only the loss rows indexed by seed_local.
        if 0 < batch_size < train_nodes.numel():
            pick = torch.randperm(train_nodes.numel(), device=train_nodes.device)[:batch_size]
            train_idx = train_nodes[pick]
        else:
            train_idx = train_nodes
        train_logits = logits[train_idx]
        if args.gnn_loss_target == "ensemble":
            loss = compute_ensemble_loss(
                train_logits, combined_ds, combined_labels, num_classes,
                task=args.gnn_task, sample_weights=sample_weights,
                seed_local=train_idx,
            )
        else:
            loss = compute_meta_loss(
                train_logits, combined_meta[train_idx],
                args.gnn_loss,
                sample_weights=sample_weights,
                train_margin=train_margin,
                seed_local=train_idx,
                focal_gamma=args.gnn_focal_gamma,
            )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ── Validate ──
        gnn.eval()
        with torch.no_grad():
            val_logits = gnn(data)[val_mask]

            if args.gnn_loss_target == "ensemble":
                val_loss = compute_ensemble_loss(
                    val_logits, combined_ds, combined_labels, num_classes,
                    task=args.gnn_task,
                    seed_local=torch.where(val_mask)[0],
                ).item()
            else:
                val_loss = compute_meta_loss(
                    val_logits, combined_meta[val_mask],
                    args.gnn_loss,
                    train_margin=train_margin,
                    seed_local=torch.where(val_mask)[0],
                    focal_gamma=args.gnn_focal_gamma,
                ).item()

            # Evaluate the prediction rule used at inference.
            if args.gnn_task == "classification":
                val_ds = combined_ds[val_mask]
                val_preds_hard = val_ds.view(-1, val_ds.size(1) // num_classes, num_classes).argmax(dim=2)
                _, val_ensemble_preds = evaluate_ensemble(
                    val_logits, val_ds, num_classes,
                    args.gnn_ens_combination_mode,
                    args.gnn_voting_weight_space,
                    hard_preds=val_preds_hard,
                )
                val_y = combined_labels[val_mask]
                val_acc = (val_ensemble_preds == val_y).float().mean().item()
                recalls = [(val_ensemble_preds[val_y == c] == c).float().mean().item()
                           for c in val_y.unique()]
                val_bacc = sum(recalls) / len(recalls) if recalls else 0.0
            else:
                val_y = combined_labels[val_mask].float()
                val_pred = evaluate_ensemble_regression(
                    val_logits,
                    combined_ds[val_mask].float(),
                    args.gnn_voting_weight_space,
                )
                val_mse = F.mse_loss(val_pred, val_y).item()
                val_rmse = val_mse ** 0.5

        scheduler.step(val_loss)

        # ── ES metric ──
        if args.gnn_es_metric == "val_acc":
            curr_metric = val_acc
        elif args.gnn_es_metric == "val_bacc":
            curr_metric = val_bacc
        else:
            curr_metric = -val_loss

        row = {"epoch": epoch, "train_loss": loss.item(), "val_loss": val_loss}
        if args.gnn_task == "classification":
            row.update(val_acc=val_acc, val_bacc=val_bacc)
        else:
            row.update(val_mse=val_mse, val_rmse=val_rmse)
        history.append(row)

        if curr_metric > best_metric + 1e-6:
            best_metric = curr_metric
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in gnn.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= args.gnn_patience:
            print(f"[GNN] Early stopping at epoch {epoch}.")
            break

        if epoch % 50 == 0 or epoch == 1:
            metric_text = (f"val_acc={val_acc:.4f}"
                           if args.gnn_task == "classification"
                           else f"val_rmse={val_rmse:.4f}")
            print(f"[GNN] Epoch {epoch}: train_loss={loss.item():.4f}, "
                  f"val_loss={val_loss:.4f}, {metric_text}")

    if best_state is not None:
        gnn.load_state_dict(best_state)
    print(f"[GNN] Best epoch: {best_epoch}, metric: {best_metric:.4f}")

    return gnn, {"history": history, "best_epoch": best_epoch, "best_metric": best_metric}


# ── Test evaluation ────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_test(
    gnn: torch.nn.Module,
    data: HeteroData,
    meta: dict,
    args,
    device: torch.device,
    fallback_model: FallbackModel | None = None,
) -> dict:
    """Evaluate the trained GNN on test data.

    Args:
        gnn: Trained GNN model.
        data: HeteroData graph (train + test).
        meta: Metadata dict from build_graph.
        args: Parsed arguments.
        device: Compute device.
        fallback_model: Optional fallback for samples with no selection.

    Returns:
        Dict of test metrics.
    """
    gnn = gnn.to(device)
    data = data.to(device)
    gnn.eval()

    combined_ds = meta["combined_ds"].to(device)
    combined_labels = meta["combined_labels"].to(device)
    test_mask = data["sample"].test_mask.bool()
    num_classes = args.num_classes

    # The concat_mlp head consumes pair features, and its first Linear was sized
    # to include them. train_gnn attaches these to its own graph; this is a
    # different graph, so without recomputing them the head is handed a tensor
    # narrower than it expects (or None, under pair_only) and dies at inference.
    if args.gnn_output_head in ("concat_mlp",):
        pair_feats = compute_pair_features(
            combined_ds, num_classes,
            pair_confidence=args.gnn_output_head_pair_confidence,
            pair_competence=args.gnn_output_head_pair_competence,
            data=data, meta=meta["combined_meta"].to(device),
        )
        if pair_feats is not None:
            data["sample"].pair_feats = pair_feats.to(device)

    logits = gnn(data)
    test_logits = logits[test_mask]

    # Apply fallback for samples with no classifier selected
    if fallback_model is not None:
        sel_sum = F.relu(test_logits).sum(dim=1)
        fb_mask = sel_sum == 0
        if fb_mask.any():
            fb_scores = fallback_model(data, test_mask)
            test_logits[fb_mask] = fb_scores[fb_mask].to(device)

    test_ds = combined_ds[test_mask]
    test_y = combined_labels[test_mask]

    results = {}
    if args.gnn_task == "classification":
        M = test_ds.size(1) // num_classes
        test_hard_preds = test_ds.view(-1, M, num_classes).argmax(dim=2)
        soft_probs, ensemble_preds = evaluate_ensemble(
            test_logits, test_ds, num_classes,
            args.gnn_ens_combination_mode,
            args.gnn_voting_weight_space,
            hard_preds=test_hard_preds,
        )

        acc = (ensemble_preds == test_y).float().mean().item()
        # Balanced accuracy
        per_class_acc = []
        for c in range(num_classes):
            mask_c = test_y == c
            if mask_c.sum() > 0:
                per_class_acc.append((ensemble_preds[mask_c] == c).float().mean().item())
        bacc = np.mean(per_class_acc) if per_class_acc else 0.0

        # Selection stats
        sel_matrix, fb_rows = compute_selection_matrix(
            test_logits, args.gnn_ens_combination_mode,
            args.gnn_voting_weight_space,
        )
        ess = (sel_matrix > 0).float().sum(dim=1).mean().item()
        fb_rate = fb_rows.float().mean().item()

        results = {
            "test_acc": acc,
            "test_bacc": bacc,
            "test_ess": ess,
            "test_fallback_rate": fb_rate,
        }
    else:
        test_preds_raw = test_ds.float()
        ensemble_pred = evaluate_ensemble_regression(
            test_logits, test_preds_raw,
            args.gnn_voting_weight_space,
        )
        mse = F.mse_loss(ensemble_pred, test_y.float()).item()
        sel_matrix, fb_rows = compute_selection_matrix(
            test_logits, args.gnn_ens_combination_mode,
            args.gnn_voting_weight_space,
        )
        results = {
            "test_mse": mse,
            "test_rmse": mse ** 0.5,
            "test_ess": (sel_matrix > 0).float().sum(dim=1).mean().item(),
            "test_fallback_rate": fb_rows.float().mean().item(),
        }

    return results


# ── Main ───────────────────────────────────────────────────────────────
