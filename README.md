# GraphRoute

GraphRoute is a graph-based dynamic ensemble selection framework. Given a pool
of candidate models, a GNN trained over a sample similarity graph learns which
models to trust for individual cases. Because different models have different
inductive biases, their reliability varies across the input space. GraphRoute
learns sample representations where proximity better reflects shared model
competence. This approach is particularly valuable for problems where models
optimized for aggregate performance tend to fail on rare edge cases, and where
failures on such cases are the most consequential.

GraphRoute operates in three stages:

1. Train a pool of diverse models.
2. Construct a graph where nodes represent samples and edges encode sample similarity.
3. Train a GNN to produce per-classifier competence scores.

- [Installation](#installation)
- [Quickstart](#quickstart)
- [Data and model interface](#data-and-model-interface)
- [Configure Experiments](#configure-experiments)
- [Configuration](#configuration)
- [Reusing a trained pool](#reusing-a-trained-pool)
- [Development](#development)

## Installation

```bash
pip install graphroute
```

GraphRoute requires Python 3.10–3.12, PyTorch 2.0 or newer, and PyTorch
Geometric 2.4 or newer.

The quickstart and experiment runner are repository-level scripts. To use them,
clone the repository and follow the [development installation](#development-installation).

## Quickstart

```bash
python quickstart.py
```

The quickstart trains a pool of four lightweight models on synthetic data, fits
a graph attention network (GAT), and compares the performance of the resulting
GNN's dynamic selection with the individual classifiers and fixed ensemble
baselines.

## Data and model interface

GraphRoute accepts PyTorch Dataset objects whose samples are (inputs, target) pairs. Datasets with
custom sample structures can provide a `collate_fn`; see the `fit_graphroute`
function documentation in
[`graphroute/run.py`](https://github.com/briannamueller/GraphRoute/blob/main/graphroute/run.py)
for the required interface. Supply the candidate model instances through the
ordered `models` list.

## Configure Experiments

[`run_experiments.py`](https://github.com/briannamueller/GraphRoute/blob/main/run_experiments.py)
reads the experiment configuration from
[`configs/experiment.yaml`](https://github.com/briannamueller/GraphRoute/blob/main/configs/experiment.yaml)
and model factories from
[`model_registry.py`](https://github.com/briannamueller/GraphRoute/blob/main/model_registry.py).
Register candidate models in `MODEL_REGISTRY`, specify their names in
`base.models`, and run:

```bash
python run_experiments.py --config configs/experiment.yaml
```

The provided YAML defines one experiment. Uncomment its optional `sweep`
section to run the Cartesian product of the listed values. Each completed
configuration and its metrics are saved as a separate JSON file under
`results/<dataset>/`. Repeating the command skips completed configurations.
Use `--force` to rerun completed configurations.

With the default `data_dir="data"`, GraphRoute reads `train.pt` and `test.pt`
from `data/<dataset>/`; each file must contain an `(inputs, targets)` tuple saved
using `torch.save`. `validation.pt` is optional. When it is absent, GraphRoute
derives validation data from the training set using `val_ratio`.

## Configuration

Every available setting and its default is defined in
[`graphroute/config.py`](https://github.com/briannamueller/GraphRoute/blob/main/graphroute/config.py).
The YAML file uses the same field names. The tables below focus on settings
whose options require an understanding of GraphRoute itself.

### General

| Argument | Meaning | Available options |
| --- | --- | --- |
| `loss_target` | Sets the GNN training objective. | `"meta_labels"`: minimizes the loss between predicted competence scores and targets that encode each model’s competence.<br>`"ensemble"`: minimizes the loss between the combined prediction and each sample’s ground-truth class label or regression target. |

### Model pool training (`base`)

| Argument | Meaning | Available options |
| --- | --- | --- |
| `base.models` | Names the ordered model pool using entries in the experiment registry. | Nonempty list of registered model names |
| `base.split_mode` | Chooses how the pool produces out-of-sample predictions for GNN training. | `"oof_stacking"`, `"split_train"` |
| `base.oof_folds` | Sets the number of folds used for OOF pool training. | Integer of at least `2` |


To use all of the training data to train both the base classifiers and the GNN without the optimistic bias caused by evaluating models on their own training samples, base.split_mode=`"oof_stacking"` uses cross-validation to generate out-of-fold predictions for GNN training. The final classifiers used for inference are then trained on the full training set.

`split_train` is less computationally expensive because each base classifier is trained only once. It divides the training data into two parts: one is used to train the base classifiers, and the other is used to train the GNN.

### Graph construction (`graph`)

<table>
  <thead>
    <tr>
      <th>Argument</th>
      <th>Meaning</th>
      <th>Available options</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>graph.node_feature_source</code></td>
      <td>Selects the sample representation supplied to the GNN.</td>
      <td rowspan="2">
        <code>"decision_space"</code>: concatenated pool predictions for the sample.<br>
        <code>"feature_space"</code>: original features (flattened if not tabular already).<br>
        <code>"embedding_mean"</code>: averages internal representation each model produces for the sample immediately before its final layer (requires same size embeddings).<br>
        <code>"embedding_concat"</code>: concatenates internal representation each model produces for the sample immediately before its final layer (embedding sizes may differ).<br>
        <code>"hybrid"</code>: decision-space representation and original features concatenated.
      </td>
    </tr>
    <tr>
      <td><code>graph.edge_feature_source</code></td>
      <td>Selects the representation used to measure similarity when constructing graph edges.</td>
    </tr>
    <tr>
      <td><code>graph.k</code></td>
      <td>Sets the number of neighbors per sample.</td>
      <td>Positive integer</td>
    </tr>
    <tr>
      <td><code>graph.neighbor_mode</code></td>
      <td>Selects ordinary nearest neighbors or class-balanced neighbors.</td>
      <td><code>"knn"</code>, <code>"class_balanced"</code></td>
    </tr>
    <tr>
      <td><code>graph.weight_mode</code></td>
      <td>Determines how sample-to-sample edge weights are calculated.</td>
      <td><code>"softmax"</code>, <code>"uniform"</code>, <code>"inverse_distance"</code>, <code>"cmdw"</code></td>
    </tr>
    <tr>
      <td><code>graph.pool_calibrate</code></td>
      <td>Enables or disables classification-pool calibration.</td>
      <td><code>True</code>, <code>False</code></td>
    </tr>
    <tr>
      <td><code>graph.calib_method</code></td>
      <td>Selects the calibration method.</td>
      <td><code>"ts-mix"</code>, <code>"logistic"</code></td>
    </tr>
  </tbody>
</table>

When oof_stacking is combined with embedding-based representations, GraphRoute uses out-of-fold predictions for GNN training, but extracts embeddings from the final base classifiers.

### GNN training and dynamic selection (`gnn`)

| Argument | Meaning | Available options |
| --- | --- | --- |
| `gnn.arch` | Selects the architecture used to learn the dynamic selection rule. | `"gat"`, `"graph_gps"`, `"mlp"` |
| `gnn.loss` | Selects the GNN training loss. | `"bce"`, `"focal_bce"`, `"soft_bce"`, `"regression"` |
| `gnn.ens_combination_mode` | Determines how model scores form the final prediction. | `"soft_weighted_voting"`, `"hard_weighted_voting"`, `"soft_voting"`, `"hard_voting"`, `"weighted_mean"` for regression |
| `gnn.voting_weight_space` | Selects how GNN scores become voting weights; when omitted, GraphRoute chooses based on `loss_target`. | `None`, `"logit"`, `"sig"` |
| `gnn.fallback` | Selects the fallback rule when no model receives a positive selection weight. | `"uniform"`, `"wacc"`, `"acc"`, `"bacc"` |


## Reusing a trained pool

GraphRoute automatically caches the results of the computation-heavy base-model training stage under pool_cache/<dataset>/pool_<pool-configuration-hash>/. The cache includes the final trained pool models and the model outputs required to construct the graph and train the GNN. The pool-configuration-hash identifies the specified model architectures and configuration for training base classifiers. Changing only the configuration for graph construction or GNN training leaves the hash unchanged, allowing the trained pool to be reused. Because `dataset` is part of the pool-cache path, use a different dataset name or delete the existing cache when the underlying data or preprocessing changes. Otherwise, GraphRoute may reuse stale models or predictions.

## Development

### Development installation

```bash
git clone https://github.com/briannamueller/GraphRoute.git
cd GraphRoute
pip install -e .
```

```bash
pytest -q
```

The quickstart and test suite run offline on CPU.

## License

MIT -- see
[LICENSE](https://github.com/briannamueller/GraphRoute/blob/main/LICENSE).
