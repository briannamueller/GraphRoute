"""Checks for the generated configuration reference."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.generate_config_reference import (
    missing_descriptions,
    render_reference,
    write_reference,
)


def test_public_configuration_fields_have_descriptions():
    assert missing_descriptions() == []


def test_generated_configuration_reference_is_current():
    assert write_reference(check=True)


def test_reference_includes_declarative_and_cross_field_constraints():
    reference = render_reference()

    assert "`graph.k` | integer | `5` | ≥ 1" in reference
    assert "`graph.pool_calibrate` and `base.weighted_by_class` are disabled" in reference
    assert "Every sweep value must be a non-empty list." in reference
