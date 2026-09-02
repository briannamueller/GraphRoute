"""Reusable base-model pools and their raw prediction outputs."""

from __future__ import annotations

import hashlib
import io
import inspect
import json
import marshal
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Optional

import torch
import torch.nn as nn

MODEL_NAME = "model_{i}.pt"
OOF_NAME = "oof_logits.pt"
MANIFEST_NAME = "manifest.json"

TrainedPool = tuple[list[nn.Module], Optional[torch.Tensor]]


def fingerprint(value: Any) -> str:
    """A stable short identity for JSON-compatible inputs."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         default=str).encode()
    return hashlib.sha256(encoded).hexdigest()[:12]


def fingerprint_model(model: nn.Module) -> str:
    """Fingerprint a complete model template plus the code its modules execute.

    Serializing the full template captures ordinary Python attributes as well as
    parameters and buffers. Hashing every distinct module class separately makes
    an implementation edit invalidate the identity even though pickle normally
    stores only a reference to that class.
    """
    hasher = hashlib.sha256()
    try:
        serialized = io.BytesIO()
        torch.save(model, serialized)
    except Exception as exc:
        raise ValueError(
            f"Cannot reliably fingerprint model template {type(model).__name__}. "
            "Use a serializable model template or disable persistent caching "
            "with cache_dir=''.") from exc
    hasher.update(serialized.getvalue())

    classes = {type(module) for module in model.modules()}
    for cls in sorted(classes, key=lambda value: (
            value.__module__, value.__qualname__)):
        hasher.update(f"{cls.__module__}.{cls.__qualname__}\0".encode())
        try:
            hasher.update(inspect.getsource(cls).encode())
        except (OSError, TypeError):
            for method_name in ("__init__", "forward"):
                method = getattr(cls, method_name, None)
                code = getattr(method, "__code__", None)
                if code is not None:
                    hasher.update(marshal.dumps(code))
    return hasher.hexdigest()[:16]


def automatic_model_ids(models: list[nn.Module]) -> list[str]:
    """Internal cache identities for an ordered list of model templates."""
    identities = []
    for model in models:
        model_type = f"{type(model).__module__}.{type(model).__qualname__}"
        model_fp = fingerprint_model(model)
        identities.append(f"{model_type}:{model_fp}")
    return identities


def fingerprint_pool(*, model_ids: list[str], base_config: dict, seed: int,
                     code_identity: dict | str | None = None,
                     model_fingerprints: list[str] | None = None) -> str:
    """Identify ordered pool members and everything that changes their weights."""
    return fingerprint({
        "model_ids": list(model_ids),
        "model_fingerprints": model_fingerprints,
        "base_config": base_config,
        "seed": seed,
        "code_identity": code_identity,
    })


def safe_component(value: str) -> str:
    """Keep a caller-supplied data identity readable and path-safe."""
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in value)
    safe = safe or "data"
    return safe if safe == value else f"{safe}-{fingerprint(value)[:8]}"


def pool_directory(cache_root: Path | str, data_id: str,
                   pool_fingerprint: str) -> Path:
    """The standalone cache location for a resolved data and pool identity."""
    return Path(cache_root) / safe_component(data_id) / f"pool_{pool_fingerprint}"


def pool_paths(directory: Path | str, n_models: int) -> list[Path]:
    """Where the pool's ``n_models`` state dicts live."""
    directory = Path(directory)
    return [directory / "models" / MODEL_NAME.format(i=i)
            for i in range(n_models)]


@dataclass
class PoolArtifact:
    """A pool whose weights and raw outputs are loaded only when needed."""

    fingerprint: str
    model_ids: tuple[str, ...]
    model_factories: tuple[Callable[[], nn.Module], ...]
    model_paths: tuple[Path, ...] = ()
    directory: Path | None = None
    output_directory: Path | None = None
    oof_path: Path | None = None
    models: list[nn.Module] | None = None
    oof_logits: torch.Tensor | None = None
    training_outputs: torch.Tensor | None = None

    def __len__(self) -> int:
        return len(self.model_ids)

    def for_data(self, *, output_directory: Path | str | None = None,
                 training_outputs: torch.Tensor | None = None) -> "PoolArtifact":
        """A view of this same pool for one dataset or federated client."""
        return replace(
            self,
            output_directory=(None if output_directory is None
                              else Path(output_directory)),
            training_outputs=training_outputs,
        )

    def load_models(self, device=None) -> list[nn.Module]:
        """Instantiate weights on CPU; collectors move one member at a time."""
        if self.models is None:
            if len(self.model_paths) != len(self.model_factories):
                raise RuntimeError("This pool has no loadable model weights.")
            loaded = []
            for factory, path in zip(self.model_factories, self.model_paths):
                model = factory()
                model.load_state_dict(torch.load(path, map_location="cpu",
                                                 weights_only=True))
                model.eval()
                loaded.append(model)
            self.models = loaded
        else:
            self.models = [m.cpu() for m in self.models]
        return self.models

    def load_oof(self) -> torch.Tensor | None:
        """Return the fold predictions created with the pool, when available."""
        if self.oof_logits is None and self.oof_path is not None:
            self.oof_logits = torch.load(self.oof_path, map_location="cpu",
                                         weights_only=True)
        return self.oof_logits

    def cached_outputs(self, name: str, loader, device, *, task: str,
                       transform: Callable[[torch.Tensor], torch.Tensor] | None = None
                       ) -> torch.Tensor:
        """Return raw pool outputs for ``loader``, computing them once if cached."""
        path = None
        if self.output_directory is not None:
            path = self.output_directory / f"{name}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            from filelock import FileLock
            with FileLock(f"{path}.lock"):
                if path.exists():
                    value = torch.load(path, map_location="cpu", weights_only=True)
                    if value.shape[0] != len(loader.dataset):
                        raise RuntimeError(
                            f"Cached outputs in {path} contain {value.shape[0]} rows, "
                            f"but the current dataset contains {len(loader.dataset)}. "
                            "Use a new dataset name or remove the stale cache.")
                    return value
                value = self._collect(loader, device, task)
                if transform is not None:
                    value = transform(value)
                _atomic_save(value, path)
                return value
        value = self._collect(loader, device, task)
        return transform(value) if transform is not None else value

    def _collect(self, loader, device, task: str) -> torch.Tensor:
        from graphroute.pool import collect_pool_predictions
        return collect_pool_predictions(self.load_models(device), loader, device,
                                        task=task).detach().cpu()


def _manifest(fingerprint_value: str, model_ids: list[str],
              has_oof: bool, data_id: str | None = None) -> dict:
    return {
        "schema_version": 2,
        "data_id": data_id,
        "fingerprint": fingerprint_value,
        "model_ids": list(model_ids),
        "has_oof": has_oof,
    }


def load_pool(directory: Path | str, model_factories: list[Callable[[], nn.Module]],
              *, fingerprint_value: str, model_ids: list[str],
              require_oof: bool = False,
              data_id: str | None = None) -> Optional[PoolArtifact]:
    """Return a complete lazy artifact, or ``None`` when it must be rebuilt."""
    directory = Path(directory)
    manifest_path = directory / MANIFEST_NAME
    if not manifest_path.exists():
        return None
    try:
        saved = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    expected = _manifest(fingerprint_value, model_ids, require_oof, data_id)
    if saved != expected:
        return None
    paths = pool_paths(directory, len(model_factories))
    if not all(path.exists() for path in paths):
        return None
    oof_path = directory / OOF_NAME
    if require_oof and not oof_path.exists():
        return None
    return PoolArtifact(
        fingerprint=fingerprint_value,
        model_ids=tuple(model_ids),
        model_factories=tuple(model_factories),
        model_paths=tuple(paths),
        directory=directory,
        output_directory=directory / "outputs",
        oof_path=oof_path if oof_path.exists() else None,
    )


def save_pool(directory: Path | str, models: list[nn.Module], *,
              fingerprint_value: str, model_ids: list[str],
              oof_logits: Optional[torch.Tensor] = None,
              data_id: str | None = None) -> PoolArtifact:
    """Publish weights, optional OOF outputs, then the validating manifest."""
    directory = Path(directory)
    (directory / "models").mkdir(parents=True, exist_ok=True)
    if oof_logits is not None:
        _atomic_save(oof_logits.detach().cpu(), directory / OOF_NAME)
    paths = pool_paths(directory, len(models))
    for model, path in zip(models, paths):
        _atomic_save(model.state_dict(), path)
    _atomic_json(_manifest(
        fingerprint_value, model_ids, oof_logits is not None, data_id),
                 directory / MANIFEST_NAME)
    return PoolArtifact(
        fingerprint=fingerprint_value,
        model_ids=tuple(model_ids),
        model_factories=(),
        model_paths=tuple(paths),
        directory=directory,
        output_directory=directory / "outputs",
        oof_path=(directory / OOF_NAME) if oof_logits is not None else None,
        models=models,
        oof_logits=None if oof_logits is None else oof_logits.detach().cpu(),
    )


def cached_pool(directory: Path | str, model_factories: list[Callable[[], nn.Module]],
                train: Callable[[], TrainedPool], *, fingerprint_value: str,
                model_ids: list[str], require_oof: bool = False,
                data_id: str | None = None) -> PoolArtifact:
    """Load a lazy pool artifact, or train and publish it once."""
    directory = Path(directory)
    from filelock import FileLock
    directory.mkdir(parents=True, exist_ok=True)
    with FileLock(directory / ".lock"):
        hit = load_pool(directory, model_factories,
                        fingerprint_value=fingerprint_value,
                        model_ids=model_ids, require_oof=require_oof,
                        data_id=data_id)
        if hit is not None:
            print(f"[Pool] cache hit {directory}")
            return hit
        print(f"[Pool] cache miss {directory}: training")
        models, oof_logits = train()
        artifact = save_pool(
            directory, models, fingerprint_value=fingerprint_value,
            model_ids=model_ids, oof_logits=oof_logits, data_id=data_id)
        artifact.model_factories = tuple(model_factories)
        return artifact


def in_memory_pool(models: list[nn.Module], *, model_ids: list[str],
                   fingerprint_value: str) -> PoolArtifact:
    """Wrap a caller-supplied pool without enabling persistent reuse."""
    if len(models) != len(model_ids):
        raise ValueError("model_ids must name every supplied pool member in order.")
    return PoolArtifact(fingerprint=fingerprint_value,
                        model_ids=tuple(model_ids), model_factories=(), models=models)


def _atomic_save(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _atomic_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)
