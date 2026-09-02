"""KNN sample-graph construction for GraphRoute."""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch_geometric.data import HeteroData
from torch_geometric.utils import to_undirected


# ── KNN Edge Building ──────────────────────────────────────────────────

def build_knn_edges(
    features: np.ndarray,
    labels: np.ndarray,
    source_indices: np.ndarray,
    destination_indices: np.ndarray,
    k: int = 5,
    neighbor_mode: str = "knn",
    weight_mode: str = "softmax",
    num_classes: int | None = None,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build sample-sample KNN edges with configurable neighbor selection and weighting.

    Args:
        features: Sample feature matrix [N, D] used for distance computation.
        labels: Targets [N]; used only by class-balanced neighbor selection.
        source_indices: Indices that can be edge sources (train samples).
        destination_indices: Indices to build edges toward (train + eval).
        k: Number of nearest neighbors (total for knn, per-class for class_balanced).
        neighbor_mode: "knn" (global top-k) or "class_balanced" (k per class).
        weight_mode: "uniform", "inverse_distance", or "softmax".
        num_classes: Number of classes for class-balanced selection.
        eps: Small constant for numerical stability.

    Returns:
        (edge_index [2, E], edge_attr [E]) as numpy arrays.
    """
    src_ids = np.asarray(source_indices, dtype=np.int64)
    dest_ids = np.asarray(destination_indices, dtype=np.int64)
    labels = np.asarray(labels)

    src_labels = labels[src_ids]
    classes = np.unique(src_labels) if neighbor_mode == "class_balanced" else np.empty(0)
    if neighbor_mode == "class_balanced" and num_classes is None:
        num_classes = len(classes)

    def _l1_dist(idx_a, idx_b):
        return np.sum(np.abs(features[idx_b] - features[idx_a]), axis=-1)

    def _select_class_balanced(j):
        sources_by_class = {c: src_ids[src_labels == c] for c in classes}
        all_neigh, all_dist = [], []
        for c in classes.tolist():
            S_c = sources_by_class[c]
            if labels[j] == c:
                S_c = S_c[S_c != j]
            if S_c.size == 0:
                continue
            d = _l1_dist(j, S_c)
            k_c = min(k, S_c.size)
            idx = np.argpartition(d, k_c - 1)[:k_c]
            all_neigh.append(S_c[idx])
            all_dist.append(d[idx])
        if not all_neigh:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
        return np.concatenate(all_neigh), np.concatenate(all_dist)

    def _select_global_knn(j):
        pool = src_ids[src_ids != j]
        if pool.size == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
        d = _l1_dist(j, pool)
        k_eff = min(k, pool.size)
        idx = np.argpartition(d, k_eff - 1)[:k_eff]
        return pool[idx], d[idx]

    def _weight_uniform(dists):
        n = dists.size
        return np.full(n, 1.0 / n, dtype=np.float32) if n > 0 else np.empty(0, dtype=np.float32)

    def _weight_inverse_distance(dists):
        inv = 1.0 / (dists + eps)
        return (inv / inv.sum()).astype(np.float32)

    def _weight_softmax(dists):
        tau = max(float(np.median(dists)), eps)
        z = -dists / tau
        z -= z.max()
        w = np.exp(z)
        return (w / w.sum()).astype(np.float32)

    weight_fn = {
        "uniform": _weight_uniform,
        "inverse_distance": _weight_inverse_distance,
        "softmax": _weight_softmax,
    }[weight_mode]

    src_list, dst_list, w_list = [], [], []
    for j in dest_ids.tolist():
        if neighbor_mode == "class_balanced":
            neigh_ids, neigh_dists = _select_class_balanced(j)
        else:
            neigh_ids, neigh_dists = _select_global_knn(j)

        if neigh_ids.size == 0:
            continue

        weights = weight_fn(neigh_dists)
        src_list.extend(neigh_ids.tolist())
        dst_list.extend([j] * neigh_ids.size)
        w_list.extend(weights.tolist())

    if not src_list:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.float32)

    edge_index = np.vstack([np.asarray(src_list, dtype=np.int64),
                            np.asarray(dst_list, dtype=np.int64)])
    edge_attr = np.asarray(w_list, dtype=np.float32)
    return edge_index, edge_attr


def build_cmdw_edges(
    features: np.ndarray,
    labels: np.ndarray,
    source_indices: np.ndarray,
    destination_indices: np.ndarray,
    k: int = 2,
    neighbor_mode: str = "class_balanced",
    num_classes: int | None = None,
    membership_mode: str = "soft",
    membership_k: int = 7,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build CMDW (Class-Membership Distance-Weighted) edges.

    Per destination, computes per-class score s_c = membership_c / (distance_c + eps),
    normalizes to class mass pi_c, then distributes within-class via softmax.

    Args:
        features: Sample feature matrix [N, D].
        labels: Integer class labels [N].
        source_indices: Train sample indices (edge sources).
        destination_indices: All sample indices (edge destinations).
        k: Neighbors per class.
        neighbor_mode: "class_balanced" or "knn".
        num_classes: Number of classes.
        membership_mode: "none", "hard", or "soft".
        membership_k: K for membership computation.
        eps: Numerical stability constant.

    Returns:
        (edge_index [2, E], edge_attr [E]) as numpy arrays.
    """
    src_ids = np.asarray(source_indices, dtype=np.int64)
    dest_ids = np.asarray(destination_indices, dtype=np.int64)
    labels = np.asarray(labels)

    src_labels = labels[src_ids]
    classes = np.unique(src_labels)
    if num_classes is None:
        num_classes = len(classes)

    mem_cache: dict[int, float] = {}

    def _membership(i):
        if membership_mode == "none":
            return 1.0
        if i in mem_cache:
            return mem_cache[i]
        xi = features[i]
        pool = src_ids[src_ids != i]
        if pool.size == 0:
            mem_cache[i] = 1.0
            return 1.0
        d = np.sum(np.abs(features[pool] - xi), axis=1)
        mk = min(membership_k, pool.size)
        nn = pool[np.argpartition(d, mk - 1)[:mk]] if mk > 0 else np.empty(0, dtype=np.int64)
        prop_same = float(np.mean(labels[nn] == labels[i])) if nn.size else 1.0
        val = 1.0 if (membership_mode == "hard" and prop_same > 0.5) else (
            prop_same if membership_mode == "soft" else 1.0)
        mem_cache[i] = val
        return val

    def _select_knn_per_class(j, q):
        per_class_n, per_class_d = {}, {}
        for c in classes.tolist():
            S_c = src_ids[src_labels == c]
            if labels[j] == c:
                S_c = S_c[S_c != j]
            if S_c.size == 0:
                continue
            d = np.sum(np.abs(features[S_c] - q), axis=1)
            k_c = min(k, S_c.size)
            idx = np.argpartition(d, k_c - 1)[:k_c]
            per_class_n[c] = S_c[idx]
            per_class_d[c] = d[idx]
        return per_class_n, per_class_d

    def _select_global_knn(j, q):
        pool = src_ids[src_ids != j]
        if pool.size == 0:
            return {}, {}
        d_all = np.sum(np.abs(features[pool] - q), axis=1)
        k_eff = min(k, pool.size)
        idx = np.argpartition(d_all, k_eff - 1)[:k_eff]
        neigh_ids = pool[idx]
        neigh_dists = d_all[idx]
        per_class_n, per_class_d = {}, {}
        for c in classes.tolist():
            mask = labels[neigh_ids] == c
            if mask.any():
                per_class_n[c] = neigh_ids[mask]
                per_class_d[c] = neigh_dists[mask]
        return per_class_n, per_class_d

    def _softmax_neg(dist):
        tau = max(float(np.median(dist)), eps)
        z = -dist / tau
        z -= z.max()
        w = np.exp(z)
        return w / w.sum()

    src_list, dst_list, w_list = [], [], []

    for j in dest_ids.tolist():
        q = features[j]
        if neighbor_mode == "class_balanced":
            per_class_n, per_class_d = _select_knn_per_class(j, q)
        else:
            per_class_n, per_class_d = _select_global_knn(j, q)

        if not per_class_n:
            continue

        # Compute per-class score: s_c = membership_c / (mean_distance_c + eps)
        class_scores = {}
        for c in per_class_n:
            m_c = np.mean([_membership(int(ni)) for ni in per_class_n[c]])
            d_c = np.mean(per_class_d[c])
            class_scores[c] = m_c / (d_c + eps)

        total_score = sum(class_scores.values())
        if total_score < eps:
            continue

        for c in per_class_n:
            pi_c = class_scores[c] / total_score
            within_w = _softmax_neg(per_class_d[c])
            for ni, wi in zip(per_class_n[c], within_w):
                src_list.append(int(ni))
                dst_list.append(j)
                w_list.append(float(pi_c * wi))

    if not src_list:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.float32)

    edge_index = np.vstack([np.asarray(src_list, dtype=np.int64),
                            np.asarray(dst_list, dtype=np.int64)])
    edge_attr = np.asarray(w_list, dtype=np.float32)
    return edge_index, edge_attr


# ── Edge Dispatch ───────────────────────────────────────────────────────

def build_edges(
    features: np.ndarray,
    labels: np.ndarray,
    n_train: int,
    n_total: int,
    k: int = 5,
    neighbor_mode: str = "knn",
    weight_mode: str = "softmax",
    num_classes: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build sample-sample edges, dispatching to KNN or CMDW builder.

    Args:
        features: Feature matrix [N_total, D] for distance computation.
        labels: Integer class labels [N_total].
        n_train: Number of training samples (first n_train rows).
        n_total: Total number of samples (train + eval).
        k: Number of neighbors.
        neighbor_mode: "knn" or "class_balanced".
        weight_mode: "uniform", "inverse_distance", "softmax", or "cmdw".
        num_classes: Number of classes.

    Returns:
        (edge_index [2, E], edge_attr [E]) as numpy arrays.
    """
    source_indices = np.arange(n_train)
    destination_indices = np.arange(n_total)

    if weight_mode == "cmdw":
        return build_cmdw_edges(
            features, labels, source_indices, destination_indices,
            k=k, neighbor_mode=neighbor_mode, num_classes=num_classes,
        )
    return build_knn_edges(
        features, labels, source_indices, destination_indices,
        k=k, neighbor_mode=neighbor_mode, weight_mode=weight_mode,
        num_classes=num_classes,
    )


# ── Bidirectionality ───────────────────────────────────────────────────

def enforce_bidirectionality(data: HeteroData) -> HeteroData:
    """Make SS edges undirected, then keep only train nodes as sources.

    This ensures validation and query nodes receive messages from training
    nodes but never send messages back to training nodes or one another.

    Args:
        data: HeteroData with sample nodes and SS edges.

    Returns:
        Modified HeteroData (in-place).
    """
    rel = ("sample", "ss", "sample")
    if rel not in data.edge_index_dict:
        return data

    num_nodes = data["sample"].x.size(0)
    ei = data[rel].edge_index
    eattr = getattr(data[rel], "edge_attr", None)

    if eattr is not None:
        ei_ud, eattr_ud = to_undirected(ei, eattr, num_nodes=num_nodes)
    else:
        ei_ud = to_undirected(ei, num_nodes=num_nodes)
        eattr_ud = None

    train_mask = getattr(data["sample"], "train_mask", None)
    if train_mask is not None:
        keep = train_mask.bool()[ei_ud[0]]  # only train samples as sources
        data[rel].edge_index = ei_ud[:, keep]
        if eattr_ud is not None:
            data[rel].edge_attr = eattr_ud[keep]
    else:
        data[rel].edge_index = ei_ud
        if eattr_ud is not None:
            data[rel].edge_attr = eattr_ud

    return data


# ── Graph Assembly ──────────────────────────────────────────────────────

def build_graph(
    train_features: torch.Tensor | np.ndarray,
    train_labels: torch.Tensor | np.ndarray,
    train_ds: torch.Tensor,
    train_meta: torch.Tensor,
    eval_features: torch.Tensor | np.ndarray | None = None,
    eval_labels: torch.Tensor | np.ndarray | None = None,
    eval_ds: torch.Tensor | None = None,
    *,
    k: int = 5,
    neighbor_mode: str = "knn",
    weight_mode: str = "softmax",
    num_classes: int | None = None,
    eval_meta: torch.Tensor | None = None,
    train_edge_features: np.ndarray | None = None,
    eval_edge_features: np.ndarray | None = None,
    eval_type: str = "val",
    task: str = "classification",
) -> Tuple[HeteroData, dict]:
    """Build the training-context KNN graph for fitting or prediction.

    Training and evaluation samples share one container, but information flow
    remains inductive: training samples are the only edge sources and all
    samples are destinations.

    Args:
        train_features: Node features for training samples [N_train, D].
        train_labels: Class labels or continuous targets [N_train].
        train_ds: Decision-space tensor for training [N_train, M*C].
        train_meta: Meta-labels for training [N_train, M].
        eval_features: Node features for eval samples [N_eval, D].
        eval_labels: Labels for eval samples [N_eval].
        eval_ds: Decision-space tensor for eval [N_eval, M*C].
        eval_meta: Meta-labels for the eval rows [N_eval, M]. Required whenever a
            validation loss is computed against these rows; defaults to zeros at
            inference, where the labels are unknown.
        k: Number of nearest neighbors.
        neighbor_mode: "knn" or "class_balanced".
        weight_mode: "uniform", "inverse_distance", "softmax", or "cmdw".
        num_classes: Number of classes (classification only).
        train_edge_features: Features for edge distance [N_train, D']. Defaults
            to the decision space when not given. Which representation these
            are is the caller's choice -- see graphroute.run.build_features.
        eval_edge_features: Separate features for edge distance [N_eval, D'].
        eval_type: "val" or "test".
        task: "classification" or "regression"; controls target dtype.

    Returns:
        (data, meta): HeteroData graph and metadata dict.
    """
    # Convert to numpy/tensor as needed
    def _to_np(x):
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            return x.numpy()
        return np.asarray(x)

    def _to_tensor(x, dtype=torch.float):
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            return x.to(dtype)
        return torch.tensor(x, dtype=dtype)

    train_features_np = _to_np(train_features)
    train_labels_np = _to_np(train_labels)

    n_train = len(train_features_np)
    has_eval = eval_features is not None

    if has_eval:
        eval_features_np = _to_np(eval_features)
        eval_labels_np = _to_np(eval_labels)
        n_eval = len(eval_features_np)
        n_total = n_train + n_eval

        combined_features = np.concatenate([train_features_np, eval_features_np], axis=0)
        combined_labels = np.concatenate([train_labels_np, eval_labels_np], axis=0)
        combined_ds = torch.cat([train_ds, eval_ds], dim=0)

        M = train_meta.size(1)
        if eval_meta is None:
            eval_meta = torch.zeros(n_eval, M)
        elif eval_meta.shape != (n_eval, M):
            raise ValueError(
                f"eval_meta must be [{n_eval}, {M}], got {tuple(eval_meta.shape)}")
        combined_meta = torch.cat([train_meta, eval_meta.float()], dim=0)
    else:
        n_total = n_train
        combined_features = train_features_np
        combined_labels = train_labels_np
        combined_ds = train_ds
        combined_meta = train_meta

    # Determine features for edge distance computation
    if train_edge_features is not None:
        if has_eval and eval_edge_features is not None:
            edge_feats = np.concatenate([
                _to_np(train_edge_features),
                _to_np(eval_edge_features),
            ], axis=0)
        else:
            edge_feats = _to_np(train_edge_features)
    else:
        edge_feats = _to_np(combined_ds)

    # Build edges
    edge_index_np, edge_attr_np = build_edges(
        edge_feats, combined_labels, n_train, n_total,
        k=k, neighbor_mode=neighbor_mode, weight_mode=weight_mode,
        num_classes=num_classes,
    )

    # Assemble HeteroData
    node_x = _to_tensor(combined_features)
    label_dtype = torch.float if task == "regression" else torch.long
    node_y = _to_tensor(combined_labels, dtype=label_dtype)

    train_mask = torch.zeros(n_total, dtype=torch.bool)
    train_mask[:n_train] = True
    eval_mask = torch.zeros(n_total, dtype=torch.bool)
    if has_eval:
        eval_mask[n_train:] = True

    data = HeteroData()
    data["sample"].x = node_x
    data["sample"].y = node_y
    data["sample"].train_mask = train_mask
    data["sample"][f"{eval_type}_mask"] = eval_mask
    data[("sample", "ss", "sample")].edge_index = torch.tensor(edge_index_np, dtype=torch.long)
    data[("sample", "ss", "sample")].edge_attr = torch.tensor(edge_attr_np, dtype=torch.float)

    # Enforce bidirectionality (train sources only)
    data = enforce_bidirectionality(data)

    meta = {
        "n_train": n_train,
        "n_total": n_total,
        "combined_ds": combined_ds,
        "combined_meta": combined_meta,
        "combined_labels": node_y,
        "eval_type": eval_type,
    }
    return data, meta
