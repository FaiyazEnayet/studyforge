"""Versioned memory models and the model registry.

The planner's default stays the exponential model. Any other model
is chosen by name and every prediction it makes is tagged with its
version, so a comparison always says which model produced each
number.
"""

from __future__ import annotations

from .base import (
    HistoryEvent,
    ModelProtocol,
    ModelState,
    Outcome,
    normalize_outcome,
)
from .exponential import ExponentialModel
from .fsrs_adapter import FSRSAdapter
from .hlr import HierarchicalHalfLifeModel

_REGISTRY: dict[str, type[ModelProtocol]] = {
    "exponential": ExponentialModel,
    "hlr": HierarchicalHalfLifeModel,
    "fsrs": FSRSAdapter,
}

_VERSIONS: dict[str, type[ModelProtocol]] = {
    cls.version: cls for cls in _REGISTRY.values()
}

DEFAULT_MODEL = "exponential"


def list_models() -> list[str]:
    """Model names in registry order (the tournament's run order)."""
    return list(_REGISTRY)


def get_model(name: str) -> ModelProtocol:
    """A fresh model instance by registry name, or ValueError."""
    try:
        return _REGISTRY[name]()
    except KeyError:
        raise ValueError(
            f"unknown model {name!r}; choose one of {list_models()}"
        ) from None


def get_model_by_version(version: str) -> ModelProtocol:
    """A fresh model instance by version tag, or ValueError.

    States store the version tag (``exponential/1.0``), while the CLI
    takes the registry name (``exponential``); this bridges the two.
    """
    cls = _VERSIONS.get(version)
    if cls is None:
        raise ValueError(
            f"unknown model version {version!r}; this build understands "
            f"{sorted(_VERSIONS)}"
        )
    return cls()


def load_model(data: dict) -> tuple[ModelProtocol, ModelState]:
    """Rebuild a model instance and its state from serialized data.

    Dispatches on the ``model`` version tag; an unknown or missing
    tag raises ValueError instead of guessing a model.
    """
    if not isinstance(data, dict):
        raise ValueError(
            f"serialized model state must be an object, got "
            f"{type(data).__name__}"
        )
    version = data.get("model")
    cls = _VERSIONS.get(version)  # type: ignore[arg-type]
    if cls is None:
        raise ValueError(
            f"unknown model version {version!r}; this build understands "
            f"{sorted(_VERSIONS)}"
        )
    model = cls.from_dict(data)
    state = ModelState(
        S=data["S"],
        last_review=data.get("last_review"),
        difficulty=data.get("difficulty", 3),
    )
    return model, state


__all__ = [
    "DEFAULT_MODEL",
    "ExponentialModel",
    "FSRSAdapter",
    "HierarchicalHalfLifeModel",
    "HistoryEvent",
    "ModelProtocol",
    "ModelState",
    "Outcome",
    "get_model",
    "get_model_by_version",
    "list_models",
    "load_model",
    "normalize_outcome",
]
