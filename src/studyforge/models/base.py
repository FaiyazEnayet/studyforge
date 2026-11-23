"""Shared types and sanity rules for the versioned model interface.

A memory model is a versioned algorithm object with three operations:

* ``predict(state, elapsed)``: recall probability after ``elapsed``
  days from the state's last review.
* ``transition(state, day, outcome)``: the state after a review on
  ``day``, using the observed outcome when one was logged and the
  model's own prediction otherwise.
* ``fit(history, as_of)``: the fitted state for one topic from its
  logged review history, using only events before ``as_of``.

Models are stateless except for fitted global parameters (the pooled
half-life in the hierarchical model). Per-topic memory lives in
``ModelState``, which models take and return as new values, so replay
is deterministic: the same history in the same order always rebuilds
the same state.

Outcome semantics
-----------------
Every logged outcome is normalized once, at the boundary, into one of
three kinds: binary (exactly 0 or 1, a right-or-wrong retrieval),
fractional (a fraction remembered, strictly between 0 and 1, or the
endpoints when the event is not marked binary), or missing (None:
the review happened but nothing was recorded, so the outcome stays
unknown and is never treated as 0). Values outside 0 to 1, booleans,
and non-numbers are rejected, not clipped, so a bad log fails loudly
instead of silently bending a fit.

Chronology guard
----------------
A fit never sees an event on or after ``as_of``; a transition never
moves to a day before the state's last review; a negative day is
rejected. A same-day repeat review is a no-op, matching the planner:
a repeat on the same day sees full recall and gains no stability.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class Outcome:
    """One normalized review outcome."""

    kind: str  # "binary", "fractional", or "missing"
    value: float | None


def normalize_outcome(value) -> Outcome:
    """Normalize a raw logged outcome, or reject it.

    Binary outcomes are exactly 0 or 1. Fractional outcomes are any
    number in 0 to 1. None means the outcome was not recorded. Anything
    else (outside 0 to 1, non-finite, a boolean, a string) raises
    ValueError naming the bad value.
    """
    if value is None:
        return Outcome(kind="missing", value=None)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"outcome must be a number between 0 and 1, got {value!r}; "
            "log the fraction actually remembered, or 0/1 for a "
            "right-or-wrong retrieval"
        )
    f = float(value)
    if not math.isfinite(f) or not 0.0 <= f <= 1.0:
        raise ValueError(
            f"outcome must be between 0 and 1, got {value!r}; log the "
            "fraction actually remembered"
        )
    if f == 0.0 or f == 1.0:
        return Outcome(kind="binary", value=f)
    return Outcome(kind="fractional", value=f)


@dataclass(frozen=True)
class HistoryEvent:
    """One logged review: the day it happened and what was observed.

    ``outcome`` is the observed recall (None when unrecorded).
    ``item`` names the retrieved item when the log is item-level;
    ``binary`` marks the outcome as a right-or-wrong retrieval rather
    than a fraction remembered.
    """

    day: int
    outcome: float | None = None
    item: str | None = None
    binary: bool = False


@dataclass(frozen=True)
class ModelState:
    """One topic's memory state under a model.

    ``S`` is the model's stability in days, in that model's own
    units (see each model for what its units mean). ``last_review``
    is the day of the most recent review, or None when the topic was
    never reviewed. ``difficulty`` is the 1 to 5 rating the default
    stability comes from.
    """

    S: float
    last_review: int | None
    difficulty: int = 3

    def __post_init__(self) -> None:
        if not isinstance(self.S, (int, float)) or isinstance(self.S, bool):
            raise ValueError(f"stability must be a number, got {self.S!r}")
        if not math.isfinite(float(self.S)) or float(self.S) <= 0:
            raise ValueError(
                f"stability must be finite and greater than 0, got "
                f"{self.S!r}"
            )
        if self.last_review is not None:
            if isinstance(self.last_review, bool) or not isinstance(
                    self.last_review, int):
                raise ValueError(
                    f"last_review must be a whole number of days, got "
                    f"{self.last_review!r}"
                )
            if self.last_review < 0:
                raise ValueError(
                    f"last_review must be 0 or later, got "
                    f"{self.last_review!r}"
                )
        if not 1 <= int(self.difficulty) <= 5:
            raise ValueError(
                f"difficulty must be between 1 and 5, got "
                f"{self.difficulty!r}"
            )


def _check_day(day: int, label: str = "day") -> int:
    if isinstance(day, bool) or not isinstance(day, int):
        raise ValueError(
            f"{label} must be a whole number of days, got {day!r}"
        )
    if day < 0:
        raise ValueError(f"{label} must be 0 or later, got {day}")
    return day


def _check_elapsed(elapsed: float) -> float:
    """Elapsed days since the last review, clamped at zero.

    A negative elapsed time means the query day precedes the last
    review; recall there is 1 by the same clamp memory.recall uses,
    so predictions stay in 0 to 1 instead of exceeding them.
    """
    if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)):
        raise ValueError(
            f"elapsed days must be a number, got {elapsed!r}"
        )
    e = float(elapsed)
    if not math.isfinite(e):
        raise ValueError(f"elapsed days must be finite, got {elapsed!r}")
    return max(0.0, e)


def _check_probability(p: float, label: str = "prediction") -> float:
    """A predicted probability must be finite and in 0 to 1."""
    if not math.isfinite(p) or not 0.0 <= p <= 1.0:
        raise ValueError(
            f"{label} must be a probability between 0 and 1, got {p!r}"
        )
    return p


def _usable_pairs(events: list[HistoryEvent]) -> list[tuple[float, float]]:
    """(interval, recall) pairs a fit can consume, in day order.

    The interval of an event is the days since the previous event;
    the first event has no interval. Only events with a recorded
    outcome and a positive interval count. Missing outcomes stay
    unknown: they advance last_review but never become a pair.
    """
    pairs: list[tuple[float, float]] = []
    prev_day: int | None = None
    for ev in events:
        if prev_day is not None and ev.outcome is not None:
            interval = ev.day - prev_day
            if interval > 0:
                pairs.append((float(interval), float(ev.outcome)))
        prev_day = ev.day
    return pairs


class ModelProtocol(ABC):
    """The interface every memory model implements.

    ``version`` names the model and its version, e.g.
    ``"exponential/1.0"``; predictions are tagged with it so a
    comparison always says which model produced each number.
    """

    version: str = "model/0.0"

    # -- core operations ------------------------------------------------

    @abstractmethod
    def predict(self, state: ModelState, elapsed_days: float) -> float:
        """Recall probability ``elapsed_days`` after the last review."""

    @abstractmethod
    def transition(
        self, state: ModelState, day: int, outcome: float | None = None
    ) -> ModelState:
        """State after a review on ``day``.

        ``outcome`` is the observed recall when one was logged, else
        the model's own prediction at that review. A review on the
        same day as the last review changes nothing; a day before the
        last review raises ValueError.
        """

    @abstractmethod
    def fit(
        self, history: list[HistoryEvent], as_of: int | None = None,
        difficulty: int = 3,
    ) -> ModelState:
        """Fitted state for one topic from its logged history.

        Only events strictly before ``as_of`` are used when ``as_of``
        is given. With too few usable pairs the default stability
        stands. ``last_review`` is the latest used event's day.
        """

    # -- stability-unit boundary ----------------------------------------

    def from_stability(
        self, S: float, last_review: int | None, difficulty: int = 3
    ) -> ModelState:
        """A StudyForge-unit stability as this model's state.

        StudyForge stability is the exponential e-folding time. Models
        whose stability means something else convert here, so planner
        code always hands models StudyForge units and gets StudyForge
        units back.
        """
        return ModelState(S=float(S), last_review=last_review,
                          difficulty=difficulty)

    def to_stability(self, state: ModelState) -> float:
        """This model's state stability back in StudyForge units."""
        return float(state.S)

    # -- versioned serialization ----------------------------------------

    def to_dict(self, state: ModelState) -> dict:
        """Serialize a fitted state, tagged with the model version."""
        return {
            "model": self.version,
            "S": float(state.S),
            "last_review": state.last_review,
            "difficulty": state.difficulty,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ModelProtocol":
        """Rebuild a model instance from serialized data.

        Only the version tag is read here; the state itself is
        rebuilt by :func:`load_model_state`. A mismatched version
        raises ValueError.
        """
        if not isinstance(data, dict) or data.get("model") != cls.version:
            raise ValueError(
                f"cannot load {cls.__name__} from data tagged "
                f"{data.get('model') if isinstance(data, dict) else data!r}; "
                f"expected {cls.version!r}"
            )
        return cls()
