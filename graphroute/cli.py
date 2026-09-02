"""Terminal overrides generated from GraphRoute's Pydantic configuration."""
from __future__ import annotations

import argparse
from types import UnionType
from typing import Literal, Union, get_args, get_origin

from pydantic import BaseModel, ValidationError

from graphroute.config import BaseConfig, GNNConfig, GraphConfig, GraphRouteConfig


def _parse_bool(value: str) -> bool:
    """Convert a terminal value such as ``true`` or ``false`` to a Boolean."""
    normalized = value.lower()
    if normalized in {"true", "yes", "1"}:
        return True
    if normalized in {"false", "no", "0"}:
        return False
    raise argparse.ArgumentTypeError(
        f"Expected true or false, received {value!r}.")


def _remove_optional(annotation):
    """Return ``T`` when a field is annotated as ``Optional[T]``."""
    origin = get_origin(annotation)
    if origin in {Union, UnionType}:
        members = [member for member in get_args(annotation)
                   if member is not type(None)]
        if len(members) == 1:
            return members[0]
    return annotation


def _argument_settings(annotation) -> dict:
    """Derive argparse settings from a Pydantic field annotation."""
    annotation = _remove_optional(annotation)
    origin = get_origin(annotation)

    if origin is Literal:
        choices = list(get_args(annotation))
        return {"choices": choices, "type": type(choices[0])}
    if origin is list:
        return {"nargs": "+", "type": _remove_optional(get_args(annotation)[0])}
    if annotation is bool:
        return {"type": _parse_bool}
    if annotation in {str, int, float}:
        return {"type": annotation}
    raise TypeError(
        f"Cannot create a terminal flag for annotation {annotation!r}.")


def _field_help(field) -> str:
    """Build help text from the description and default in config.py."""
    parts = []
    if field.description:
        parts.append(field.description)
    if field.is_required():
        parts.append("Required.")
    else:
        default = field.get_default(call_default_factory=True)
        parts.append(f"Default: {default!r}.")
    return " ".join(parts)


def _add_fields(
    argument_group: argparse._ArgumentGroup,
    model: type[BaseModel],
    *,
    prefix: str = "",
    excluded: set[str] | None = None,
) -> None:
    """Create terminal flags from one Pydantic configuration model."""
    excluded = excluded or set()
    for field_name, field in model.model_fields.items():
        if field_name in excluded:
            continue

        path = f"{prefix}_{field_name}" if prefix else field_name
        flag = f"--{path.replace('_', '-')}"
        destination = f"{prefix}__{field_name}" if prefix else field_name
        settings = _argument_settings(field.annotation)
        if "choices" not in settings:
            settings["metavar"] = field_name.upper()
        argument_group.add_argument(
            flag,
            dest=destination,
            default=argparse.SUPPRESS,
            help=_field_help(field),
            **settings,
        )


def build_parser() -> argparse.ArgumentParser:
    """Build terminal flags from the GraphRoute Pydantic configuration."""
    parser = argparse.ArgumentParser(
        description="Run a GraphRoute experiment.")

    general = parser.add_argument_group("General")
    _add_fields(
        general, GraphRouteConfig, excluded={"base", "graph", "gnn"})
    _add_fields(
        parser.add_argument_group("Base-model pool"), BaseConfig, prefix="base")
    _add_fields(
        parser.add_argument_group("Graph construction"), GraphConfig,
        prefix="graph")
    _add_fields(
        parser.add_argument_group("GNN training and dynamic selection"),
        GNNConfig, prefix="gnn")
    return parser


def _set_nested_value(values: dict, destination: str, value) -> None:
    """Apply a parsed terminal value to its nested configuration group."""
    path = destination.split("__")
    current = values
    for part in path[:-1]:
        current = current.setdefault(part, {})
    current[path[-1]] = value


def config_from_args(
    argv: list[str] | None = None,
    *,
    base: GraphRouteConfig | None = None,
) -> GraphRouteConfig:
    """Apply explicitly supplied terminal overrides to a configuration."""
    parser = build_parser()
    parsed = parser.parse_args(argv)
    # Pydantic supplies every value that neither the script nor the terminal set.
    values = {} if base is None else base.model_dump(exclude_unset=True)
    for destination, value in vars(parsed).items():
        _set_nested_value(values, destination, value)

    try:
        return GraphRouteConfig.model_validate(values)
    except ValidationError as error:
        parser.error(str(error))


__all__ = ["build_parser", "config_from_args"]
