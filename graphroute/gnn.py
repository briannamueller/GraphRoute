"""GNN architectures and output heads for GraphRoute."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATv2Conv


# ── Helpers ─────────────────────────────────────────────────────────────

def _drop_edges_with_attr(
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor | None,
    p: float,
    training: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """DropEdge that also filters the corresponding edge_attr tensor."""
    if not training or p <= 0.0:
        return edge_index, edge_attr
    num_edges = edge_index.size(1)
    keep = torch.rand(num_edges, device=edge_index.device) >= p
    filtered_index = edge_index[:, keep]
    filtered_attr = edge_attr[keep] if edge_attr is not None else None
    return filtered_index, filtered_attr


# ── Output Heads ────────────────────────────────────────────────────────

class LinearHead(nn.Module):
    """Linear(h_sample) -> [N, M]. Ignores pool-member embeddings."""

    def __init__(self, hidden_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, out_dim)

    def forward(self, sample_emb: torch.Tensor,
                clf_emb: torch.Tensor | None = None,
                pair_feats: torch.Tensor | None = None) -> torch.Tensor:
        return self.linear(sample_emb)


class DotLearnedHead(nn.Module):
    """Cosine-similarity head with a learnable pool-member embedding table.

    ``logits[n, j] = temperature * cosine(h_sample[n], clf_emb[j]) + bias[j]``

    Uses a separate learnable parameter table [M, hidden_dim] trained
    end-to-end (CLIP-style learned temperature, clamped to [1, 100]).
    """

    _LOGIT_SCALE_MAX = 4.6052  # ln(100)

    def __init__(self, hidden_dim: int, out_dim: int):
        super().__init__()
        self.score_clf_emb = nn.Parameter(torch.empty(out_dim, hidden_dim))
        nn.init.normal_(self.score_clf_emb, std=0.02)
        self.clf_bias = nn.Parameter(torch.zeros(out_dim))
        init_val = torch.tensor(2.6593)  # ln(1/0.07), CLIP default
        self.logit_scale = nn.Parameter(init_val)

    def forward(self, sample_emb: torch.Tensor,
                clf_emb: torch.Tensor | None = None,
                pair_feats: torch.Tensor | None = None) -> torch.Tensor:
        s = F.normalize(sample_emb, dim=-1)
        c = F.normalize(self.score_clf_emb, dim=-1)
        scale = self.logit_scale.clamp(max=self._LOGIT_SCALE_MAX).exp()
        return scale * (s @ c.T) + self.clf_bias.unsqueeze(0)


class ConcatMLPLearnedHead(nn.Module):
    """MLP([h_sample || learned_clf_emb || pair_feats]) -> scalar per pair.

    Uses a learnable pool-member embedding table [M, hidden_dim].

    Args:
        hidden_dim: Embedding dimension for samples and pool members.
        out_dim: Number of pool members M.
        normalize: Apply LayerNorm to embeddings before concatenation.
        pair_feat_dim: Extra per-(sample, model) feature dimension.
        pair_only: Omit the model embedding; input is [h_sample || pair_feats].
    """

    def __init__(self, hidden_dim: int, out_dim: int, *,
                 normalize: bool = False,
                 pair_feat_dim: int = 0, pair_only: bool = False):
        super().__init__()
        self.pair_feat_dim = int(pair_feat_dim)
        self.pair_only = pair_only
        if pair_only:
            self.score_clf_emb = None
            mlp_input_dim = hidden_dim + self.pair_feat_dim
        else:
            self.score_clf_emb = nn.Parameter(torch.empty(out_dim, hidden_dim))
            nn.init.normal_(self.score_clf_emb, std=0.02)
            mlp_input_dim = 2 * hidden_dim + self.pair_feat_dim
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.clf_bias = nn.Parameter(torch.zeros(out_dim))
        self.normalize = normalize
        if normalize:
            self.sample_norm = nn.LayerNorm(hidden_dim)
            if not pair_only:
                self.clf_norm = nn.LayerNorm(hidden_dim)

    def forward(self, sample_emb: torch.Tensor,
                clf_emb: torch.Tensor | None = None,
                pair_feats: torch.Tensor | None = None) -> torch.Tensor:
        N, D = sample_emb.shape
        if self.normalize:
            sample_emb = self.sample_norm(sample_emb)

        if self.pair_only:
            M = pair_feats.shape[1]
            s_exp = sample_emb.unsqueeze(1).expand(N, M, D)
            combined = torch.cat([s_exp, pair_feats], dim=2)  # [N, M, D + P]
        else:
            c = self.score_clf_emb
            if self.normalize:
                c = self.clf_norm(c)
            M = c.shape[0]
            s_exp = sample_emb.unsqueeze(1).expand(N, M, D)
            c_exp = c.unsqueeze(0).expand(N, M, D)
            combined = torch.cat([s_exp, c_exp], dim=2)  # [N, M, 2D]
            if pair_feats is not None:
                combined = torch.cat([combined, pair_feats], dim=2)

        scores = self.mlp(combined).squeeze(-1)  # [N, M]
        return scores + self.clf_bias.unsqueeze(0)


def build_output_head(mode: str, hidden_dim: int, out_dim: int, *,
                      normalize: bool = False,
                      pair_feat_dim: int = 0,
                      pair_only: bool = False) -> nn.Module:
    """Factory for output head modules.

    Args:
        mode: "linear", "dot", or "concat_mlp".
        hidden_dim: Embedding dimension.
        out_dim: Number of pool members M.
        normalize: L2-normalize (dot) or LayerNorm (concat_mlp) before scoring.
        pair_feat_dim: Per-(sample, model) feature dimension for concat_mlp.
        pair_only: concat_mlp only: omit the model embedding column.
    """
    mode = mode.lower()
    if mode == "dot":
        return DotLearnedHead(hidden_dim, out_dim)
    if mode == "concat_mlp":
        return ConcatMLPLearnedHead(
            hidden_dim, out_dim,
            normalize=normalize,
            pair_feat_dim=pair_feat_dim,
            pair_only=pair_only,
        )
    return LinearHead(hidden_dim, out_dim)


# ── GNN Architectures ──────────────────────────────────────────────────

class _TrainMemoryGPSConv(nn.Module):
    """GPS block whose global channel can read only training nodes.

    PyG's ``GPSConv`` performs global self-attention over every node in the
    graph. GraphRoute graphs also contain validation or query nodes, so that
    behavior makes predictions depend on the other examples evaluated in the
    same call. Here the local channel remains a GAT over ``edge_index``, while
    the global channel is standard cross-attention: every node is a query, and
    training nodes alone provide keys and values.

    Applying this restriction at every layer keeps training representations
    independent of evaluation data and makes each evaluation prediction
    independent of the other evaluation rows.
    """

    def __init__(self, *, channels: int, conv: nn.Module, heads: int,
                 dropout: float) -> None:
        super().__init__()
        self.conv = conv
        self.dropout = float(dropout)
        self.attn = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.local_norm = nn.LayerNorm(channels)
        self.global_norm = nn.LayerNorm(channels)
        self.output_norm = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 2, channels),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                train_mask: torch.Tensor, *,
                edge_attr: torch.Tensor | None = None) -> torch.Tensor:
        train_mask = train_mask.bool()
        if train_mask.ndim != 1 or train_mask.numel() != x.size(0):
            raise ValueError(
                "train_mask must be one-dimensional with one entry per node.")
        if not bool(train_mask.any()):
            raise ValueError("GraphGPS needs at least one training node.")

        local = self.conv(x, edge_index, edge_attr=edge_attr)
        local = F.dropout(local, p=self.dropout, training=self.training)
        local = self.local_norm(local + x)

        # MultiheadAttention permits different query and source lengths. The
        # residual retains each query's own representation; only its shared
        # context is restricted to the fixed training memory.
        queries = x.unsqueeze(0)
        memory = x[train_mask].unsqueeze(0)
        global_context, _ = self.attn(
            queries, memory, memory, need_weights=False,
        )
        global_context = global_context.squeeze(0)
        global_context = F.dropout(
            global_context, p=self.dropout, training=self.training)
        global_context = self.global_norm(global_context + x)

        out = local + global_context
        return self.output_norm(out + self.mlp(out))


class SampleGAT(nn.Module):
    """GATv2 over the sample-sample KNN graph.

    Message passing uses GATv2Conv with multi-head attention.
    Output is one score per sample and pool member.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        heads: int = 4,
        feat_dropout: float = 0.2,
        attn_dropout: float = 0.2,
        edge_dropout: float = 0.0,
        out_dim: int = 1,
        use_sample_residual: bool = False,
        use_edge_attr: bool = False,
        concat: bool = False,
        output_head_mode: str = "linear",
        output_head_norm: bool = False,
        pair_feat_dim: int = 0,
        pair_only: bool = False,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(feat_dropout)
        self.edge_dropout = float(edge_dropout)
        self.activation = nn.ReLU()
        self.use_sample_residual = bool(use_sample_residual)
        self.use_edge_attr = bool(use_edge_attr)
        self.concat = bool(concat)

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        edge_dim = 1 if self.use_edge_attr else None

        self.convs = nn.ModuleList()
        for layer_i in range(num_layers):
            is_last = (layer_i == num_layers - 1)
            do_concat = self.concat and not is_last
            in_dim = hidden_dim * heads if (self.concat and layer_i > 0) else hidden_dim
            self.convs.append(
                GATv2Conv(
                    in_dim, hidden_dim,
                    heads=heads, concat=do_concat, dropout=attn_dropout,
                    add_self_loops=True, edge_dim=edge_dim,
                )
            )

        if output_head_mode in ("dot", "concat_mlp"):
            self.output_head = build_output_head(
                output_head_mode, hidden_dim, out_dim,
                normalize=output_head_norm,
                pair_feat_dim=pair_feat_dim,
                pair_only=pair_only,
            )
            self.sample_head = None
        else:
            self.sample_head = nn.Linear(hidden_dim, out_dim)
            self.output_head = None

    def forward(self, data: HeteroData) -> torch.Tensor:
        x = self.input_proj(data["sample"].x)
        sample_residual = x if self.use_sample_residual else None

        edge_index = data[("sample", "ss", "sample")].edge_index
        edge_attr = None
        if self.use_edge_attr:
            edge_attr = getattr(data[("sample", "ss", "sample")], "edge_attr", None)
            if edge_attr is not None and edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)

        for layer_idx, conv in enumerate(self.convs):
            ei, ea = _drop_edges_with_attr(edge_index, edge_attr, self.edge_dropout, self.training)
            x = conv(x, ei, edge_attr=ea)
            if layer_idx != len(self.convs) - 1:
                x = self.dropout(self.activation(x))
            else:
                x = self.activation(x)

        if sample_residual is not None:
            x = x + sample_residual

        if self.output_head is not None:
            pair_feats = getattr(data["sample"], "pair_feats", None)
            if pair_feats is not None:
                return self.output_head(x, pair_feats=pair_feats)
            return self.output_head(x)
        return self.sample_head(x)


class SampleGraphGPS(nn.Module):
    """Inductive GraphGPS with local GATv2 and train-memory attention.

    Evaluation nodes receive local messages and global context from training
    nodes, but never provide context to training nodes or one another.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        heads: int = 4,
        feat_dropout: float = 0.2,
        attn_dropout: float = 0.2,
        edge_dropout: float = 0.0,
        out_dim: int = 1,
        use_sample_residual: bool = False,
        use_edge_attr: bool = False,
        output_head_mode: str = "linear",
        output_head_norm: bool = False,
        pair_feat_dim: int = 0,
        pair_only: bool = False,
    ) -> None:
        super().__init__()
        self.edge_dropout = float(edge_dropout)
        self.use_sample_residual = bool(use_sample_residual)
        self.use_edge_attr = bool(use_edge_attr)

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        edge_dim = 1 if self.use_edge_attr else None

        self.gps_layers = nn.ModuleList()
        for _ in range(num_layers):
            local_conv = GATv2Conv(
                hidden_dim, hidden_dim,
                heads=heads, concat=False, dropout=attn_dropout,
                add_self_loops=True, edge_dim=edge_dim,
            )
            gps = _TrainMemoryGPSConv(
                channels=hidden_dim,
                conv=local_conv,
                heads=heads,
                dropout=feat_dropout,
            )
            self.gps_layers.append(gps)

        if output_head_mode in ("dot", "concat_mlp"):
            self.output_head = build_output_head(
                output_head_mode, hidden_dim, out_dim,
                normalize=output_head_norm,
                pair_feat_dim=pair_feat_dim,
                pair_only=pair_only,
            )
            self.sample_head = None
        else:
            self.sample_head = nn.Linear(hidden_dim, out_dim)
            self.output_head = None

    def forward(self, data: HeteroData) -> torch.Tensor:
        x = self.input_proj(data["sample"].x)
        sample_residual = x if self.use_sample_residual else None

        ss_rel = ("sample", "ss", "sample")
        if ss_rel in data.edge_index_dict:
            edge_index = data[ss_rel].edge_index
            edge_attr = None
            if self.use_edge_attr:
                edge_attr = getattr(data[ss_rel], "edge_attr", None)
                if edge_attr is not None and edge_attr.dim() == 1:
                    edge_attr = edge_attr.view(-1, 1)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=x.device)
            edge_attr = None

        train_mask = getattr(data["sample"], "train_mask", None)
        if train_mask is None:
            raise ValueError("GraphGPS requires data['sample'].train_mask.")

        for gps_layer in self.gps_layers:
            ei, ea = _drop_edges_with_attr(edge_index, edge_attr, self.edge_dropout, self.training)
            x = gps_layer(x, ei, train_mask, edge_attr=ea)

        if sample_residual is not None:
            x = x + sample_residual

        if self.output_head is not None:
            pair_feats = getattr(data["sample"], "pair_feats", None)
            if pair_feats is not None:
                return self.output_head(x, pair_feats=pair_feats)
            return self.output_head(x)
        return self.sample_head(x)


class SampleMLP(nn.Module):
    """Feedforward MLP baseline (no graph structure).

    Uses sample node features only — ignores edges. Acts as an ablation
    baseline to measure the contribution of message passing.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        feat_dropout: float = 0.2,
        out_dim: int = 1,
        output_head_mode: str = "linear",
        output_head_norm: bool = False,
        pair_feat_dim: int = 0,
        pair_only: bool = False,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(feat_dropout),
        ]
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(feat_dropout),
            ])

        if output_head_mode in ("dot", "concat_mlp"):
            self.net = nn.Sequential(*layers)
            self.output_head = build_output_head(
                output_head_mode, hidden_dim, out_dim,
                normalize=output_head_norm,
                pair_feat_dim=pair_feat_dim,
                pair_only=pair_only,
            )
            self.sample_head = None
        else:
            layers.append(nn.Linear(hidden_dim, out_dim))
            self.net = nn.Sequential(*layers)
            self.output_head = None
            self.sample_head = None

    def forward(self, data: HeteroData) -> torch.Tensor:
        x = self.net(data["sample"].x)
        if self.output_head is not None:
            pair_feats = getattr(data["sample"], "pair_feats", None)
            if pair_feats is not None:
                return self.output_head(x, pair_feats=pair_feats)
            return self.output_head(x)
        return x


def build_gnn(arch: str, **kwargs) -> nn.Module:
    """Factory to instantiate a GNN architecture by name.

    Args:
        arch: One of "gat", "graph_gps", "mlp".
        **kwargs: Forwarded to the architecture constructor.
    """
    if arch == "gat":
        return SampleGAT(**kwargs)
    if arch == "graph_gps":
        return SampleGraphGPS(**kwargs)
    if arch == "mlp":
        return SampleMLP(**kwargs)
    raise ValueError(f"Unknown GNN architecture: {arch!r}")
