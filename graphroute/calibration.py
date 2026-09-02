"""Post-hoc probability calibration for the base-classifier pool.

A calibrator is fit on one set of logits and applied to others:

    calibrator = get_calibrator("ts-mix", num_classes=C).fit(logits, labels)
    probs = calibrator.predict_proba(other_logits)

``logits`` is ``[N, C]`` and the result is ``[N, C]`` probabilities. The two
methods are selected by ``graph.calib_method``:

``ts-mix``
    Temperature scaling. One scalar per classifier; rescales confidence without
    reordering predictions.

``logistic``
    Platt scaling for two classes, structured matrix scaling for more. Both can
    reorder predictions.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
import torchmin

#: A log-probability below this is indistinguishable from log(0) in float32.
_LOG_TINY = float(np.log(np.finfo(np.float32).tiny))

#: The methods ``get_calibrator`` accepts.
METHODS = ("ts-mix", "logistic")


def _log_of(probs: torch.Tensor) -> torch.Tensor:
    """Probabilities as logits, floored so a zero does not become ``-inf``."""
    return torch.clamp(torch.log(probs.float()), min=_LOG_TINY)


def _normalized(logits: torch.Tensor) -> torch.Tensor:
    """Logits as bounded log-probabilities.

    Every estimator here is invariant to a per-row shift, so normalizing costs
    nothing and buys a guarantee: an unnormalized logit can be arbitrarily
    large, while a log-probability is bounded above by zero and floored below.
    """
    return _log_of(torch.softmax(logits, dim=-1))


class TemperatureScaler:
    """Temperature scaling, mixed with a uniform distribution.

    The inverse temperature is a single scalar, fit by bisection on the
    derivative of the mean cross-entropy -- which is monotone in that scalar,
    so bisection is exact rather than a search that might not converge.

    The fitted probabilities are then mixed with a uniform distribution at
    weight ``1 / (N + 1)``. Without it a confident classifier produces exact
    zeros, and log loss is infinite there. The weight vanishes as the
    calibration set grows, so it costs nothing where it is not needed.

    Reference:
        Guo, Pleiss, Sun and Weinberger. On calibration of modern neural
        networks. ICML 2017.
    """

    def __init__(self, steps: int = 30, log_lo: float = -16.0, log_hi: float = 16.0):
        self.steps = steps
        self.log_lo, self.log_hi = log_lo, log_hi
        self.inv_temp_, self.n_fit_ = 1.0, 0

    def _ce_derivative(self, inv_temp: float, logits: torch.Tensor,
                       labels: torch.Tensor) -> float:
        """d/d(inv_temp) of the mean cross-entropy at this inverse temperature."""
        probs = torch.softmax(inv_temp * logits, dim=-1)
        return (torch.mean(torch.sum(logits * probs, dim=-1))
                - torch.mean(logits[torch.arange(logits.shape[0]), labels])).item()

    def fit(self, logits: torch.Tensor, labels: torch.Tensor) -> "TemperatureScaler":
        """Fit the inverse temperature. Args: logits [N, C], labels [N]."""
        lo, hi = self.log_lo, self.log_hi
        for _ in range(self.steps):                  # bisect in log-space, so
            mid = lo + 0.5 * (hi - lo)               # the scalar stays positive
            if self._ce_derivative(math.exp(mid), logits, labels) > 0:
                hi = mid
            else:
                lo = mid
        self.inv_temp_ = math.exp(0.5 * (lo + hi))
        self.n_fit_ = logits.shape[0]
        return self

    def predict_proba(self, logits: torch.Tensor) -> torch.Tensor:
        """Calibrated probabilities [N, C]."""
        probs = torch.softmax(self.inv_temp_ * logits, dim=-1)
        weight = 1.0 / (self.n_fit_ + 1)
        return (1.0 - weight) * probs + weight / probs.shape[-1]


class PlattScaler:
    """Platt scaling: ``sigmoid(b + w * logit)``, two classes only.

    An affine map of the binary logit, fit by unpenalized logistic regression.
    Unlike temperature scaling the intercept lets it shift the decision
    threshold, which is what an imbalanced pool member usually needs.

    Reference:
        Platt. Probabilistic outputs for support vector machines. Advances in
        Large Margin Classifiers, 1999.
    """

    def __init__(self, max_iter: int = 200):
        self.max_iter = max_iter
        self.bias_, self.weight_ = 0.0, 1.0

    @staticmethod
    def _binary_logit(logits: torch.Tensor) -> torch.Tensor:
        """The single logit behind a two-column score, as [N]."""
        return (logits[:, 1] - logits[:, 0]).float()

    def fit(self, logits: torch.Tensor, labels: torch.Tensor) -> "PlattScaler":
        """Fit intercept and slope. Args: logits [N, 2], labels [N] in {0, 1}."""
        if logits.shape[-1] != 2:
            raise ValueError(
                f"Platt scaling is binary; got {logits.shape[-1]} classes. Use "
                f"structured matrix scaling for more.")
        x = self._binary_logit(logits)
        y = labels.to(x.dtype)

        def objective(params):
            return F.binary_cross_entropy_with_logits(params[0] + params[1] * x, y)

        result = torchmin.minimize(
            objective, torch.zeros(2, dtype=x.dtype), method="bfgs",
            options={"max_iter": self.max_iter})
        self.bias_, self.weight_ = result.x[0].item(), result.x[1].item()
        return self

    def predict_proba(self, logits: torch.Tensor) -> torch.Tensor:
        """Calibrated probabilities [N, 2]."""
        p = torch.sigmoid(self.bias_ + self.weight_ * self._binary_logit(logits))
        return torch.stack([1.0 - p, p], dim=1)


class StructuredMatrixScaler:
    """Structured matrix scaling: ``softmax((I + dW) x + b)`` on scaled logits.

    Temperature scaling is applied first and its scalar held fixed, so ``dW``
    and ``b`` only have to describe what a single temperature could not. The
    penalty is separate for the intercept, the diagonal of ``dW`` and its
    off-diagonal, each scaled by ``k**rho / n**tau`` -- a matrix has ``k**2``
    parameters, so without a penalty that grows with the class count it fits
    the calibration set rather than calibrating.

    Reference:
        Berta, Holzmuller, Jordan and Bach. Structured matrix scaling for
        multi-class calibration. AISTATS 2026.
    """

    def __init__(self, rho: float = 1.0, tau: float = 1.0,
                 lambda_intercept: float = 1.0, lambda_diagonal: float = 1.0,
                 lambda_off_diagonal: float = 1.0):
        self.rho, self.tau = rho, tau
        self.lambda_intercept = lambda_intercept
        self.lambda_diagonal = lambda_diagonal
        self.lambda_off_diagonal = lambda_off_diagonal

    def _scaled_log_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """The temperature-scaled log-probabilities the matrix acts on.

        The temperature step consumes and produces probabilities, so its output
        goes back through ``_log_of`` -- not ``_normalized``, which would treat
        those probabilities as logits and squash them a second time.
        """
        return _log_of(self.temperature_.predict_proba(_normalized(logits)))

    def fit(self, logits: torch.Tensor, labels: torch.Tensor) -> "StructuredMatrixScaler":
        """Fit the scaling matrix. Args: logits [N, C], labels [N]."""
        n, k = logits.shape
        self.temperature_ = TemperatureScaler().fit(_normalized(logits), labels)
        x, y = self._scaled_log_probs(logits), labels.long()

        reg_intercept = self.lambda_intercept * k ** self.rho / n ** self.tau
        reg_diagonal = self.lambda_diagonal * k ** self.rho / n ** self.tau
        reg_off_diagonal = (self.lambda_off_diagonal
                            * (k * (k - 1)) ** self.rho / n ** self.tau)

        def objective(params):
            delta, bias = params[:k * k].view(k, k), params[k * k:]
            loss = F.cross_entropy(x + F.linear(x, delta, bias), y)
            diagonal = delta.diagonal()
            return (loss
                    + reg_intercept * bias.pow(2).sum()
                    + reg_diagonal * diagonal.pow(2).sum()
                    + reg_off_diagonal * (delta.pow(2).sum() - diagonal.pow(2).sum()))

        start = torch.zeros(k * (k + 1), dtype=x.dtype)
        result = torchmin.minimize(
            objective, start, method="l-bfgs" if start.numel() > 1000 else "bfgs")

        # Carry the intercept as a final column and append a constant 1 to the
        # inputs, so applying the calibrator is one matrix multiply.
        matrix = torch.eye(k, dtype=x.dtype) + result.x[:k * k].view(k, k)
        self.matrix_ = torch.hstack([matrix, result.x[k * k:].unsqueeze(1)])
        return self

    def predict_proba(self, logits: torch.Tensor) -> torch.Tensor:
        """Calibrated probabilities [N, C]."""
        x = self._scaled_log_probs(logits)
        x = torch.hstack([x, torch.ones(len(x), 1, dtype=x.dtype)])
        return torch.softmax(x @ self.matrix_.T, dim=-1)


def get_calibrator(method: str = "ts-mix", num_classes: int = 2):
    """Build an unfitted calibrator.

    Args:
        method: One of ``METHODS``.
        num_classes: Decides the ``logistic`` estimator -- Platt scaling is
            defined for two classes and structured matrix scaling generalizes
            it, so the class count selects rather than the caller.

    Returns:
        A calibrator with ``fit(logits, labels)`` and ``predict_proba(logits)``.
    """
    if method == "ts-mix":
        return TemperatureScaler()
    if method == "logistic":
        return PlattScaler() if num_classes == 2 else StructuredMatrixScaler()
    raise ValueError(f'Unknown calibration method "{method}". '
                     f'Known: {", ".join(METHODS)}.')
