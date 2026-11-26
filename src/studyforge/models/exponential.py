"""The unchanged current model, behind the versioned interface.

This is the StudyForge baseline exactly as it ships today: recall
``R(t) = exp(-t / S)`` with stability S in days (the exponential
e-folding time), the SM-2 style review update ``S' = S * (1 + GROWTH
* (1 - R))``, and per-topic stability fitting by least squares on log
recall. Nothing about its behavior changes here; it only gains a
version tag (``exponential/1.0``), a ``predict`` / ``transition`` /
``fit`` surface, and versioned serialization, so the planner and the
tournament can treat it as one model among several.
"""

from __future__ import annotations

import math

from ..memory import GROWTH, fit_stability, initial_stability
from .base import (
    HistoryEvent,
    ModelProtocol,
    ModelState,
    _check_day,
    _check_elapsed,
    _check_probability,
    _usable_pairs,
    normalize_outcome,
)


class ExponentialModel(ModelProtocol):
    """Ebbinghaus exponential forgetting with the SM-2 style update."""

    version = "exponential/1.0"

    def predict(self, state: ModelState, elapsed_days: float) -> float:
        if state.last_review is None:
            # Never studied: nothing learned, recall is zero.
            return 0.0
        elapsed = _check_elapsed(elapsed_days)
        return _check_probability(math.exp(-elapsed / state.S))

    def transition(
        self, state: ModelState, day: int, outcome: float | None = None
    ) -> ModelState:
        day = _check_day(day)
        if state.last_review is not None:
            if day < state.last_review:
                raise ValueError(
                    f"review on day {day} comes before the last review "
                    f"on day {state.last_review}; history must run "
                    "forward in time"
                )
            if day == state.last_review:
                # Same-day repeat review: recall is 1, stability gains
                # nothing, so the state is unchanged.
                return state
            elapsed = day - state.last_review
        else:
            # First review is full learning.
            elapsed = 0
        if outcome is None:
            R = 0.0 if state.last_review is None else self.predict(
                state, elapsed)
        else:
            R = normalize_outcome(outcome).value
            assert R is not None
            if R == 0.0:
                # An observed zero is a failed retrieval, not learning:
                # it changes neither stability nor the last review day.
                # A predicted zero (outcome None) keeps the documented
                # rule: a first review is acquisition, and a planned
                # review uses the model's own prediction.
                return state
        new_S = state.S * (1.0 + GROWTH * (1.0 - R))
        return ModelState(S=new_S, last_review=day,
                          difficulty=state.difficulty)

    def fit(
        self, history: list[HistoryEvent], as_of: int | None = None,
        difficulty: int = 3,
    ) -> ModelState:
        events = _fit_events(history, as_of)
        pairs = _usable_pairs(events)
        S_fit = fit_stability(pairs)
        S = initial_stability(difficulty) if S_fit is None else S_fit
        last = events[-1].day if events else None
        return ModelState(S=S, last_review=last, difficulty=difficulty)


def _fit_events(
    history: list[HistoryEvent], as_of: int | None
) -> list[HistoryEvent]:
    """History sorted by day, using only events before ``as_of``.

    Sorting is stable, so replay is deterministic for any input
    order. Events on or after ``as_of`` are excluded, never fitted.
    """
    if as_of is not None:
        as_of = _check_day(as_of, "as_of")
    events = sorted(history, key=lambda e: e.day)
    for ev in events:
        _check_day(ev.day, "review day")
        if ev.outcome is not None:
            normalize_outcome(ev.outcome)
    if as_of is not None:
        events = [e for e in events if e.day < as_of]
    return events


class DefaultExponentialModel(ExponentialModel):
    """The unfitted default: difficulty stability, never fitted.

    Predicts and transitions exactly like the exponential model,
    but ``fit`` always returns the difficulty default stability, so
    a comparison against the fitted exponential isolates what the
    fitting itself adds.
    """

    version = "exponential-default/1.0"

    def fit(
        self, history: list[HistoryEvent], as_of: int | None = None,
        difficulty: int = 3,
    ) -> ModelState:
        events = _fit_events(history, as_of)
        last = events[-1].day if events else None
        return ModelState(
            S=initial_stability(difficulty), last_review=last,
            difficulty=difficulty,
        )
