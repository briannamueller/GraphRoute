"""Validated configuration for GraphRoute."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

Task = Literal["classification", "regression"]
FeatureSource = Literal["decision_space", "feature_space", "embedding_mean",
                        "embedding_concat", "hybrid"]


class BaseConfig(BaseModel):
    """Base-model pool training."""
    model_config = {"extra": "forbid"}

    models: Optional[list[str]] = Field(
        default=None,
        description=(
            "Names identifying the ordered base-model pool. Built-in names from "
            "graphroute.models are used when model instances are not supplied."
        ),
    )
    # oof_stacking gives every row an out-of-sample pool prediction. split_train
    # is cheaper, but the pool is trained on only part of the available data.
    split_mode: Literal["oof_stacking", "split_train"] = Field(
        default="oof_stacking",
        description=(
            "Chooses whether GNN training uses out-of-fold pool predictions or "
            "a separate pool-training split."
        ),
    )
    oof_folds: int = Field(
        default=5,
        ge=2,
        description="Number of folds used when split_mode is oof_stacking.",
    )
    es_metric: Literal["val_loss", "val_acc", "val_bacc"] = "val_loss"
    es_patience: int = Field(20, ge=1)
    lr: float = Field(5e-4, gt=0)
    optimizer: Literal["SGD", "Adam"] = "Adam"
    weight_decay: float = Field(5e-4, ge=0)
    weighted_by_class: bool = True
    epochs: int = Field(300, ge=1)
    batch_size: int = Field(10, ge=1)


class GraphConfig(BaseModel):
    """Sample graph construction."""
    model_config = {"extra": "forbid"}

    node_feature_source: FeatureSource = Field(
        default="decision_space",
        description="Sample representation supplied to the GNN as node features.",
    )
    edge_feature_source: FeatureSource = Field(
        default="decision_space",
        description="Sample representation used to measure similarity for graph edges.",
    )
    pool_calibrate: bool = Field(
        default=True,
        description="Whether to calibrate classification-pool predictions.",
    )
    calib_method: Literal["ts-mix", "logistic"] = Field(
        default="ts-mix",
        description="Calibration method used when pool calibration is enabled.",
    )
    k: int = Field(
        default=5,
        ge=1,
        description="Number of graph neighbors per sample.",
    )
    neighbor_mode: Literal["knn", "class_balanced"] = Field(
        default="knn",
        description="Uses ordinary nearest neighbors or class-balanced neighbors.",
    )
    weight_mode: Literal["softmax", "uniform", "inverse_distance", "cmdw"] = Field(
        default="softmax",
        description="Determines how sample-to-sample edge weights are calculated.",
    )


class GNNConfig(BaseModel):
    """GNN meta-learner and ensemble rule."""
    model_config = {"extra": "forbid"}

    arch: Literal["gat", "graph_gps", "mlp"] = Field(
        default="gat",
        description="Architecture used for the GraphRoute meta-learner.",
    )
    hidden_dim: int = Field(128, ge=1)
    layers: int = Field(2, ge=1)
    heads: int = Field(4, ge=1)
    concat: bool = False
    use_sample_residual: bool = False
    use_edge_attr: bool = False

    output_head: Literal["linear", "dot", "concat_mlp"] = "linear"
    output_head_norm: bool = False
    pair_confidence: bool = False
    pair_competence: Literal["none", "gain"] = "none"
    pair_only: bool = False

    feat_dropout: float = Field(0.2, ge=0, lt=1)
    attn_dropout: float = Field(0.2, ge=0, lt=1)
    edge_dropout: float = Field(0.0, ge=0, lt=1)

    lr: float = Field(5e-4, gt=0)
    weight_decay: float = Field(1e-4, ge=0)
    epochs: int = Field(300, ge=1)
    patience: int = Field(20, ge=1)
    # 0 computes loss over every training node; a positive value samples nodes.
    batch_size: int = Field(0, ge=0)
    es_metric: Literal["val_loss", "val_acc", "val_bacc"] = "val_loss"

    loss: Literal["bce", "focal_bce", "soft_bce", "regression"] = Field(
        default="bce",
        description="Loss function used to train the GNN.",
    )
    focal_gamma: float = Field(2.0, ge=0)
    sample_weight_mode: Literal["none", "class_prevalence", "difficulty"] = "none"

    ens_combination_mode: Literal["soft_weighted_voting", "hard_weighted_voting",
                                  "soft_voting", "hard_voting",
                                  "weighted_mean"] = Field(
        default="soft_weighted_voting",
        description="Determines how model scores form the combined prediction.",
    )
    voting_weight_space: Optional[Literal["logit", "sig"]] = Field(
        default=None,
        description=(
            "Transforms GNN scores into voting weights; when omitted, GraphRoute "
            "chooses based on loss_target."
        ),
    )
    fallback: Literal["uniform", "wacc", "acc", "bacc"] = Field(
        default="uniform",
        description="Fallback rule used when no model has a positive selection weight.",
    )


class GraphRouteConfig(BaseModel):
    """Everything needed for one run."""
    model_config = {"extra": "forbid"}

    task: Task = "classification"
    loss_target: Literal["meta_labels", "ensemble"] = Field(
        default="meta_labels",
        description="Sets the GNN training objective.",
    )
    dataset: str = Field(
        min_length=1,
        description="Dataset name used for data loading and persistent pool reuse.",
    )
    data_dir: str = Field(
        default="data",
        description=(
            "Parent directory containing the dataset folder."
        ),
    )
    num_classes: int = Field(10, ge=1)
    device: Literal["cpu", "cuda", "mps", "auto"] = "auto"
    seed: int = 0
    val_ratio: float = Field(
        default=0.25,
        gt=0,
        lt=1,
        description=(
            "Fraction of training data used for validation when validation.pt is absent."
        ),
    )

    base: BaseConfig = Field(default_factory=BaseConfig)
    graph: GraphConfig = Field(default_factory=GraphConfig)
    gnn: GNNConfig = Field(default_factory=GNNConfig)

    @model_validator(mode="after")
    def _regression_excludes_class_notions(self):
        """Resolve regression defaults and reject classification-only settings."""
        if self.task == "regression":
            if "num_classes" in self.model_fields_set and self.num_classes != 1:
                raise ValueError(
                    "Scalar regression uses one model output; omit num_classes or "
                    "set num_classes=1.")
            self.num_classes = 1

            if ("pool_calibrate" in self.graph.model_fields_set
                    and self.graph.pool_calibrate):
                raise ValueError(
                    "graph.pool_calibrate calibrates class probabilities and is not "
                    "defined for regression. Set it to false.")
            self.graph.pool_calibrate = False

            if ("weighted_by_class" in self.base.model_fields_set
                    and self.base.weighted_by_class):
                raise ValueError(
                    "base.weighted_by_class is classification-only. Set it to false.")
            self.base.weighted_by_class = False

            if self.graph.neighbor_mode == "class_balanced":
                raise ValueError(
                    "graph.neighbor_mode='class_balanced' balances a sample's "
                    "neighbours across classes, which a regression target does not "
                    "have. Use 'knn'.")
            if self.graph.weight_mode == "cmdw":
                raise ValueError(
                    "graph.weight_mode='cmdw' uses class membership and is not "
                    "defined for regression.")
            if self.base.es_metric != "val_loss":
                raise ValueError(
                    "Regression base models support early stopping on val_loss.")
            if self.gnn.es_metric != "val_loss":
                raise ValueError(
                    "Regression GNN training supports early stopping on val_loss.")
            if self.gnn.fallback != "uniform":
                raise ValueError(
                    "Regression currently uses uniform averaging when no model is "
                    "selected; accuracy-based fallback modes are classification-only.")
            if self.gnn.pair_confidence:
                raise ValueError(
                    "gnn.pair_confidence contains class probabilities and is not "
                    "defined for regression.")
            if self.gnn.sample_weight_mode != "none":
                raise ValueError(
                    "Regression currently supports gnn.sample_weight_mode='none'; "
                    "the other modes are defined from classes or class confidence.")
            if ("ens_combination_mode" in self.gnn.model_fields_set
                    and self.gnn.ens_combination_mode != "weighted_mean"):
                raise ValueError(
                    "Regression combines predictions by a weighted mean; set "
                    "gnn.ens_combination_mode='weighted_mean'.")
            self.gnn.ens_combination_mode = "weighted_mean"
            if self.loss_target == "meta_labels" and self.gnn.loss == "soft_bce":
                raise ValueError(
                    "gnn.loss='soft_bce' uses classification margins. Use 'bce' or "
                    "'regression' for regression competence targets.")
        elif self.gnn.ens_combination_mode == "weighted_mean":
            raise ValueError(
                "gnn.ens_combination_mode='weighted_mean' is for regression; "
                "classification requires a voting mode.")
        return self

    def resolved_voting_weight_space(self) -> str:
        """Ensemble-mode training weights pool members by sigmoid(logits), so
        scoring in "sig" space applies the same function at inference. "logit"
        (relu) is a different, unbounded one that over-weights confident
        pool members relative to what was optimised."""
        if self.gnn.voting_weight_space is not None:
            return self.gnn.voting_weight_space
        return "sig" if self.loss_target == "ensemble" else "logit"

    def gnn_namespace(self) -> SimpleNamespace:
        """Build the flat settings object used by training and evaluation."""
        g = self.gnn
        return SimpleNamespace(
            num_classes=self.num_classes,
            gnn_task=self.task,
            gnn_loss_target=self.loss_target,
            gnn_arch=g.arch,
            gnn_loss=g.loss,
            gnn_focal_gamma=g.focal_gamma,
            gnn_sample_weight_mode=g.sample_weight_mode,
            gnn_output_head=g.output_head,
            gnn_output_head_pair_confidence=g.pair_confidence,
            gnn_output_head_pair_competence=g.pair_competence,
            gnn_lr=g.lr,
            gnn_weight_decay=g.weight_decay,
            gnn_epochs=g.epochs,
            gnn_patience=g.patience,
            gnn_es_metric=g.es_metric,
            gnn_batch_size=g.batch_size,
            gnn_ens_combination_mode=g.ens_combination_mode,
            gnn_voting_weight_space=self.resolved_voting_weight_space(),
        )

    def pair_feat_dim(self) -> int:
        """Width of the per-(sample, model) features the concat_mlp head takes.

        compute_pair_features stacks the model's probability vector
        (num_classes wide) when pair_confidence is on, and a scalar neighbourhood
        gain when pair_competence is "gain". The head's first Linear is sized
        from this, so getting it wrong is a shape error at the first forward.
        """
        g = self.gnn
        if g.output_head != "concat_mlp":
            return 0
        return (self.num_classes if g.pair_confidence else 0) + (g.pair_competence == "gain")

    def validate_head(self) -> None:
        """pair_only leaves the head with nothing but pair features to read."""
        g = self.gnn
        if g.output_head == "concat_mlp" and g.pair_only and self.pair_feat_dim() == 0:
            raise ValueError(
                "gnn.pair_only drops the model embedding, so the head reads only "
                "pair features -- but neither pair_confidence nor pair_competence is "
                "enabled, leaving nothing to read. Enable one, or set pair_only=False.")

    def gnn_kwargs(self, input_dim: int, out_dim: int, pair_feat_dim: int = 0) -> dict:
        """Constructor arguments for :func:`graphroute.gnn.build_gnn`.

        Filtered to what the chosen architecture actually accepts: the MLP has no
        attention heads and GraphGPS no head-concatenation, so passing every
        setting to every architecture makes two of the three fail to construct.
        """
        import inspect

        from graphroute.gnn import build_gnn

        g = self.gnn
        candidate = dict(
            input_dim=input_dim, out_dim=out_dim,
            hidden_dim=g.hidden_dim, num_layers=g.layers, heads=g.heads,
            feat_dropout=g.feat_dropout, attn_dropout=g.attn_dropout,
            edge_dropout=g.edge_dropout, concat=g.concat,
            use_sample_residual=g.use_sample_residual, use_edge_attr=g.use_edge_attr,
            output_head_mode=g.output_head, output_head_norm=g.output_head_norm,
            pair_feat_dim=pair_feat_dim, pair_only=g.pair_only,
        )
        cls = build_gnn.__globals__[
            {"gat": "SampleGAT", "graph_gps": "SampleGraphGPS", "mlp": "SampleMLP"}[g.arch]]
        accepted = set(inspect.signature(cls.__init__).parameters)
        return {k: v for k, v in candidate.items() if k in accepted}
