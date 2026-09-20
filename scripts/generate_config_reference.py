"""Generate GraphRoute's exhaustive configuration reference from Pydantic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pydantic import BaseModel

from graphroute.config import (
    BaseConfig,
    GNNConfig,
    GraphConfig,
    GraphRouteConfig,
    GraphRouteExperimentConfig,
)


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATH = ROOT / "docs" / "reference" / "configuration.md"
CONFIG_MODELS = (
    BaseConfig,
    GraphConfig,
    GNNConfig,
    GraphRouteConfig,
    GraphRouteExperimentConfig,
)
_MISSING = object()


def _resolve(schema: dict, root: dict) -> dict:
    """Resolve one local JSON Schema reference."""
    if "$ref" not in schema:
        return schema
    target = root
    for part in schema["$ref"].removeprefix("#/").split("/"):
        target = target[part]
    return target


def _type_name(schema: dict, root: dict) -> str:
    """Translate JSON Schema types into compact reader-facing names."""
    if "$ref" in schema:
        return _resolve(schema, root).get("title", "mapping")
    if "anyOf" in schema:
        names = []
        for member in schema["anyOf"]:
            name = _type_name(member, root)
            if name not in names:
                names.append(name)
        return " | ".join(names)
    kind = schema.get("type")
    if kind == "array":
        return f"list[{_type_name(schema.get('items', {}), root)}]"
    if kind == "object":
        values = schema.get("additionalProperties")
        if isinstance(values, dict):
            return f"mapping[string, {_type_name(values, root)}]"
        return "mapping"
    return {
        "integer": "integer",
        "number": "number",
        "boolean": "boolean",
        "string": "string",
        "null": "null",
    }.get(kind, "value")


def _code(value) -> str:
    text = value if isinstance(value, str) else json.dumps(
        value, sort_keys=True, separators=(",", ":")
    )
    return f"`{text.replace('|', '&#124;')}`"


def _constraints(schema: dict) -> str:
    """Render declarative Pydantic constraints from JSON Schema."""
    parts = []
    if "enum" in schema:
        parts.append(", ".join(_code(value) for value in schema["enum"]))
    if "const" in schema:
        parts.append(_code(schema["const"]))
    for key, label in (
        ("minimum", "≥"),
        ("exclusiveMinimum", ">"),
        ("maximum", "≤"),
        ("exclusiveMaximum", "<"),
    ):
        if key in schema:
            parts.append(f"{label} {schema[key]}")
    if "minLength" in schema:
        length = schema["minLength"]
        parts.append("non-empty" if length == 1 else f"at least {length} characters")
    if "maxLength" in schema:
        parts.append(f"at most {schema['maxLength']} characters")
    if "minItems" in schema:
        count = schema["minItems"]
        parts.append("non-empty" if count == 1 else f"at least {count} items")
    if "maxItems" in schema:
        count = schema["maxItems"]
        parts.append(f"at most {count} item" + ("" if count == 1 else "s"))
    if "anyOf" in schema:
        choices = []
        for member in schema["anyOf"]:
            rendered = _constraints(member)
            if rendered != "—" and rendered not in choices:
                choices.append(rendered)
        if choices:
            parts.append(" or ".join(choices))
    return "; ".join(parts) or "—"


def _defaults(model: type[BaseModel]) -> dict:
    """Read defaults, including values supplied by default factories."""
    defaults = {}
    for name, field in model.model_fields.items():
        if field.is_required():
            continue
        value = field.get_default(call_default_factory=True)
        if isinstance(value, BaseModel):
            value = value.model_dump(mode="json", by_alias=True)
        defaults[field.alias or name] = value
    return defaults


def _rows(
    model: type[BaseModel],
    prefix: str = "",
    *,
    exclude: set[str] | None = None,
    defaults: dict | None = None,
) -> list[list[str]]:
    """Flatten a Pydantic model's JSON Schema into reference-table rows."""
    root = model.model_json_schema(by_alias=True)
    excluded = exclude or set()
    default_values = _defaults(model) if defaults is None else defaults
    rows: list[list[str]] = []

    def visit(node: dict, path: str, values=_MISSING, required: bool = False):
        resolved = _resolve(node, root)
        properties = resolved.get("properties")
        if properties:
            required_names = set(resolved.get("required", []))
            current = values if isinstance(values, dict) else {}
            for name, child in properties.items():
                if path == prefix and name in excluded:
                    continue
                visit(
                    child,
                    f"{path}.{name}" if path else name,
                    current.get(name, _MISSING),
                    name in required_names,
                )
            return

        default = values
        if default is _MISSING:
            default = resolved.get("default", _MISSING)
        default_text = "required" if required and default is _MISSING else "—"
        if default is not _MISSING:
            default_text = _code(default)
        description = resolved.get("description", "—").replace("\n", " ")
        rows.append(
            [
                f"`{path}`",
                _type_name(node, root).replace("|", "\\|"),
                default_text,
                _constraints(resolved),
                description.replace("|", "\\|"),
            ]
        )

    visit(root, prefix, default_values)
    return rows


def _table(rows: list[list[str]]) -> str:
    lines = [
        "| Setting | Type | Default | Allowed | Description |",
        "|---|---|---|---|---|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def _section(
    title: str,
    model: type[BaseModel],
    prefix: str = "",
    *,
    exclude: set[str] | None = None,
) -> str:
    return f"## {title}\n\n{_table(_rows(model, prefix, exclude=exclude))}"


def missing_descriptions() -> list[str]:
    """Return public configuration fields without reference descriptions."""
    return [
        f"{model.__name__}.{name}"
        for model in CONFIG_MODELS
        for name, field in model.model_fields.items()
        if not field.description
    ]


def render_reference() -> str:
    """Render the complete configuration reference."""
    parts = [
        "# Configuration reference\n",
        _section(
            "General settings",
            GraphRouteConfig,
            exclude={"base", "graph", "gnn"},
        ),
        _section("Base-model pool settings", BaseConfig, "base"),
        _section("Graph construction settings", GraphConfig, "graph"),
        _section("GNN and ensemble settings", GNNConfig, "gnn"),
        _section("Experiment-file settings", GraphRouteExperimentConfig),
        """## Compatibility

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
""",
    ]
    return "\n".join(parts).rstrip() + "\n"


def write_reference(*, check: bool = False) -> bool:
    """Write the reference, or report whether the committed copy is current."""
    content = render_reference()
    if REFERENCE_PATH.exists() and REFERENCE_PATH.read_text() == content:
        return True
    if check:
        print("Generated configuration reference is stale.")
        print("Run: python scripts/generate_config_reference.py")
        return False
    REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    REFERENCE_PATH.write_text(content)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    raise SystemExit(0 if write_reference(check=args.check) else 1)


if __name__ == "__main__":
    main()
