# Configuration reference

## General settings

| Setting | Type | Default | Allowed | Description |
|---|---|---|---|---|
| `loss_target` | string | `meta_labels` | `meta_labels`, `ensemble` | Sets the GNN training objective. |
| `task` | string | `classification` | `classification`, `regression` | Prediction task. |
| `dataset` | string | required | non-empty | Dataset name used for data loading and persistent pool reuse. |
| `data_dir` | string | `data` | — | Parent directory containing the dataset folder. |
| `num_classes` | integer | `10` | ≥ 1 | Number of classes; scalar regression uses one output. |
| `device` | string | `auto` | `cpu`, `cuda`, `mps`, `auto` | Device used for training and evaluation. |
| `seed` | integer | `0` | — | Random seed. |
| `val_ratio` | number | `0.25` | > 0; < 1 | Fraction of training data used for validation when validation.pt is absent. |

## Base-model pool settings

| Setting | Type | Default | Allowed | Description |
|---|---|---|---|---|
| `base.models` | list[string] \| null | `null` | — | Names identifying the ordered base-model pool. Built-in names from graphroute.models are used when model instances are not supplied. |
| `base.split_mode` | string | `oof_stacking` | `oof_stacking`, `split_train` | Chooses whether GNN training uses out-of-fold pool predictions or a separate pool-training split. |
| `base.oof_folds` | integer | `5` | ≥ 2 | Number of folds used when split_mode is oof_stacking. |
| `base.es_metric` | string | `val_loss` | `val_loss`, `val_acc`, `val_bacc` | Validation metric used to select checkpoints. |
| `base.es_patience` | integer | `20` | ≥ 1 | Epochs without improvement before stopping. |
| `base.lr` | number | `0.0005` | > 0 | Optimizer learning rate. |
| `base.optimizer` | string | `Adam` | `SGD`, `Adam` | Optimizer used to train base models. |
| `base.weight_decay` | number | `0.0005` | ≥ 0 | Optimizer weight decay. |
| `base.weighted_by_class` | boolean | `true` | — | Weight classification loss by class frequency. |
| `base.epochs` | integer | `300` | ≥ 1 | Maximum training epochs. |
| `base.batch_size` | integer | `10` | ≥ 1 | Training batch size. |

## Graph construction settings

| Setting | Type | Default | Allowed | Description |
|---|---|---|---|---|
| `graph.node_feature_source` | string | `decision_space` | non-empty | Sample representation supplied to the GNN as node features. |
| `graph.edge_feature_source` | string | `decision_space` | non-empty | Sample representation used to measure similarity for graph edges. |
| `graph.embedding_normalization` | string | `none` | `none`, `per_model_l2` | Optional normalization applied to each pool-model embedding. |
| `graph.distance_metric` | string | `manhattan` | `manhattan`, `cosine` | Distance used to select graph neighbors. |
| `graph.pool_calibrate` | boolean | `true` | — | Whether to calibrate classification-pool predictions. |
| `graph.calib_method` | string | `ts-mix` | `ts-mix`, `logistic` | Calibration method used when pool calibration is enabled. |
| `graph.k` | integer | `5` | ≥ 1 | Number of graph neighbors per sample. |
| `graph.neighbor_mode` | string | `knn` | `knn`, `class_balanced` | Uses ordinary nearest neighbors or class-balanced neighbors. |
| `graph.weight_mode` | string | `softmax` | `softmax`, `uniform`, `inverse_distance`, `cmdw` | Determines how sample-to-sample edge weights are calculated. |

## GNN and ensemble settings

| Setting | Type | Default | Allowed | Description |
|---|---|---|---|---|
| `gnn.arch` | string | `gat` | `gat`, `hetero_gat`, `graph_gps`, `mlp` | Architecture used for the GraphRoute meta-learner. |
| `gnn.hidden_dim` | integer | `128` | ≥ 1 | Hidden embedding width. |
| `gnn.layers` | integer | `2` | ≥ 1 | Number of GNN layers. |
| `gnn.heads` | integer | `4` | ≥ 1 | Attention heads per GAT layer. |
| `gnn.concat` | boolean | `false` | — | Concatenate heads in intermediate GAT layers. |
| `gnn.use_sample_residual` | boolean | `false` | — | Add input features to the final sample embeddings. |
| `gnn.use_edge_attr` | boolean | `false` | — | Use graph edge weights as attention features. |
| `gnn.output_head` | string | `linear` | `linear`, `dot`, `concat_mlp` | Head used to score each sample-model pair. |
| `gnn.output_head_norm` | boolean | `false` | — | Normalize embeddings before output scoring. |
| `gnn.pair_confidence` | boolean | `false` | — | Include model probabilities in concat_mlp features. |
| `gnn.pair_competence` | string | `none` | `none`, `gain` | Optional neighborhood-gain feature for concat_mlp. |
| `gnn.pair_only` | boolean | `false` | — | Omit model embeddings from concat_mlp features. |
| `gnn.feat_dropout` | number | `0.2` | ≥ 0; < 1 | Feature-dropout probability. |
| `gnn.attn_dropout` | number | `0.2` | ≥ 0; < 1 | Attention-dropout probability. |
| `gnn.edge_dropout` | number | `0.0` | ≥ 0; < 1 | Edge-dropout probability. |
| `gnn.lr` | number | `0.0005` | > 0 | Optimizer learning rate. |
| `gnn.weight_decay` | number | `0.0001` | ≥ 0 | Optimizer weight decay. |
| `gnn.epochs` | integer | `300` | ≥ 1 | Maximum training epochs. |
| `gnn.patience` | integer | `20` | ≥ 1 | Epochs without improvement before stopping. |
| `gnn.batch_size` | integer | `0` | ≥ 0 | Training nodes per batch; zero uses every node. |
| `gnn.es_metric` | string | `val_loss` | `val_loss`, `val_acc`, `val_bacc` | Validation metric used to select checkpoints. |
| `gnn.loss` | string | `bce` | `bce`, `focal_bce`, `soft_bce`, `regression` | Loss function used to train the GNN. |
| `gnn.focal_gamma` | number | `2.0` | ≥ 0 | Focusing parameter for focal BCE loss. |
| `gnn.sample_weight_mode` | string | `none` | `none`, `class_prevalence`, `difficulty` | Training-node weighting scheme. |
| `gnn.ens_combination_mode` | string | `soft_weighted_voting` | `soft_weighted_voting`, `hard_weighted_voting`, `soft_voting`, `hard_voting`, `weighted_mean` | Determines how model scores form the combined prediction. |
| `gnn.voting_weight_space` | string \| null | `null` | `logit`, `sig`, `dense` | Transforms GNN scores into voting weights; when omitted, GraphRoute chooses based on loss_target. 'dense' uses every sigmoid-transformed score without thresholding. |
| `gnn.fallback` | string | `uniform` | `uniform`, `wacc`, `acc`, `bacc` | Fallback rule used when no model has a positive selection weight. |

## Experiment-file settings

| Setting | Type | Default | Allowed | Description |
|---|---|---|---|---|
| `sweep` | mapping[string, list[value]] | `{}` | — | Cartesian parameter axes. |

## Compatibility

### Regression

- Set `num_classes` to `1` or omit it.
- `graph.pool_calibrate` and `base.weighted_by_class` are disabled and cannot be
  set to `true`.
- `gnn.ens_combination_mode` defaults to `weighted_mean`; no other value is
  supported.
- Regression does not support `gnn.arch: hetero_gat`,
  `graph.neighbor_mode: class_balanced`, `graph.weight_mode: cmdw`,
  `gnn.pair_confidence: true`, or a non-`none` `gnn.sample_weight_mode`.
- Use `val_loss` for both early-stopping metrics and `uniform` for
  `gnn.fallback`.
- With `loss_target: meta_labels`, regression does not support
  `gnn.loss: soft_bce`.
- Classification does not support `gnn.ens_combination_mode: weighted_mean`.

### Output-head compatibility

When `gnn.output_head` is `concat_mlp` and `gnn.pair_only` is `true`, enable
`gnn.pair_confidence` or set `gnn.pair_competence` to `gain`.

### Sweeps

Every sweep value must be a non-empty list. A sweep path may name a general
setting such as `seed`, or one field under `base`, `graph`, or `gnn`, such as
`graph.k`. Deeper and unknown paths are rejected.
