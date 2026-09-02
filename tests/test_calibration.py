"""Pool calibration: both methods, binary and multiclass."""

import pytest
import torch

from graphroute.calibration import (METHODS, PlattScaler, StructuredMatrixScaler,
                                    TemperatureScaler, get_calibrator)
from graphroute.pool import apply_calibrators, calibrate_pool


def _logits(n=36, m=2, c=3, seed=0):
    return torch.randn(n, m, c, generator=torch.Generator().manual_seed(seed))


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("num_classes", [2, 3, 7])
def test_calibrating_a_pool_gives_distributions_that_reapply(method, num_classes):
    logits = _logits(c=num_classes)
    labels = torch.arange(36) % num_classes

    probabilities, calibrators = calibrate_pool(logits, labels, method)
    reapplied = apply_calibrators(calibrators, logits)

    assert probabilities.shape == (36, 2, num_classes)
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(36, 2), atol=1e-5)
    assert (probabilities > 0).all()
    # Fitting and applying are the same operation, so a calibrator carried to a
    # later phase must reproduce what calibration already returned.
    assert torch.allclose(probabilities, reapplied)


@pytest.mark.parametrize("method", METHODS)
def test_each_pool_member_is_calibrated_independently(method):
    # One member's logits are scaled up, so a shared correction cannot fit both.
    logits = _logits(m=2, c=3)
    logits[:, 1, :] *= 6.0
    labels = torch.arange(36) % 3

    _, calibrators = calibrate_pool(logits, labels, method)
    assert len(calibrators) == 2
    assert calibrators[0] is not calibrators[1]


def test_the_class_count_selects_the_logistic_estimator():
    assert isinstance(get_calibrator("logistic", num_classes=2), PlattScaler)
    assert isinstance(get_calibrator("logistic", num_classes=5), StructuredMatrixScaler)
    assert isinstance(get_calibrator("ts-mix", num_classes=5), TemperatureScaler)


def test_an_unknown_method_names_the_known_ones():
    with pytest.raises(ValueError, match="ts-mix"):
        get_calibrator("nonesuch")


def test_platt_scaling_rejects_more_than_two_classes():
    with pytest.raises(ValueError, match="binary"):
        PlattScaler().fit(torch.randn(20, 3), torch.arange(20) % 3)


def test_temperature_scaling_cools_an_overconfident_classifier():
    # Logits far larger than the labels justify: the fitted scalar must shrink
    # them, and calibrated confidence must drop.
    generator = torch.Generator().manual_seed(0)
    logits = torch.randn(500, 2, generator=generator) * 8.0
    labels = torch.randint(0, 2, (500,), generator=generator)

    scaler = TemperatureScaler().fit(logits, labels)
    assert scaler.inv_temp_ < 1.0
    assert scaler.predict_proba(logits).max() < torch.softmax(logits, dim=-1).max()


def test_the_uniform_mixture_keeps_probabilities_off_zero():
    # Separable logits: without the mixture the losing class underflows to 0.
    logits = torch.tensor([[-400.0, 400.0]] * 8)
    labels = torch.ones(8, dtype=torch.long)

    probabilities = TemperatureScaler().fit(logits, labels).predict_proba(logits)
    assert (probabilities > 0).all()
    assert torch.isfinite(torch.log(probabilities)).all()


def test_calibration_recovers_a_known_temperature():
    # Draw labels from probabilities at a known temperature, then check the
    # fitted scalar undoes it. Recovery is what "calibrated" has to mean.
    generator = torch.Generator().manual_seed(0)
    true_logits = torch.randn(20000, 3, generator=generator)
    labels = torch.multinomial(torch.softmax(true_logits, dim=-1), 1,
                               generator=generator).squeeze(1)

    scaler = TemperatureScaler().fit(true_logits * 4.0, labels)   # 4x overconfident
    assert scaler.inv_temp_ == pytest.approx(0.25, rel=0.1)


def test_logistic_calibration_beats_the_raw_pool_on_log_loss():
    # A miscalibrated, class-shifted classifier: a scalar cannot fix the shift,
    # which is the case the logistic estimators exist for.
    generator = torch.Generator().manual_seed(0)
    logits = torch.randn(4000, 4, generator=generator)
    labels = torch.multinomial(torch.softmax(logits, dim=-1), 1,
                               generator=generator).squeeze(1)
    observed = logits * 3.0 + torch.tensor([2.0, 0.0, -1.0, 0.0])

    def log_loss(probabilities):
        return -torch.log(probabilities[torch.arange(len(labels)), labels]).mean()

    calibrated = get_calibrator("logistic", num_classes=4).fit(observed, labels)
    assert log_loss(calibrated.predict_proba(observed)) < log_loss(
        torch.softmax(observed, dim=-1))
