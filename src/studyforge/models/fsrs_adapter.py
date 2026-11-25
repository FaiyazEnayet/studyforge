"""Transparent FSRS-family reference adapter.

FSRS retrievability has the power-law form (FSRS-4.5/5/6)::

    R(t) = (1 + FACTOR * t / S)^(-DECAY)

with FACTOR = 19/81 and DECAY = 0.5. FSRS stability S is the time
until retrievability falls to 90%: at t = S the formula gives exactly
(100/81)^(-1/2) = 0.9.

StudyForge stability is a different unit: the exponential e-folding
time, where R = 1/e. The two are never interchanged. The conversion
is exact and explicit: the time for the exponential curve to fall to
90% is ``S_sf * ln(10/9)``, so::

    S_fsrs = S_sf * ln(10/9)
    S_sf   = S_fsrs / ln(10/9)

This adapter is a reference baseline, not a full FSRS implementation:
it isolates the retrievability *shape* (power law vs exponential)
while reusing the StudyForge review update, mapped through the unit
conversion. It fits only item-level binary outcomes, the data FSRS
is defined on; topic-level fractional recalls are refused rather
than silently fitted as if they were per-item grades.
"""

from __future__ import annotations

import math

from ..memory import fit_stability
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
from .exponential import _fit_events

# Published FSRS retrievability constants (FSRS-4.5/5/6).
FSRS_FACTOR = 19.0 / 81.0
FSRS_DECAY = 0.5

# ln(10/9): the exponential curve's time to 90% per unit of S.
_LN_10_OVER_9 = math.log(10.0 / 9.0)


def from_studyforge_stability(S_sf: float) -> float:
    """StudyForge e-folding stability as an FSRS (time-to-90%) stability."""
    if not math.isfinite(S_sf) or S_sf <= 0:
        raise ValueError(
            f"StudyForge stability must be finite and greater than 0, "
            f"got {S_sf!r}"
        )
    return S_sf * _LN_10_OVER_9


def to_studyforge_stability(S_fsrs: float) -> float:
    """An FSRS (time-to-90%) stability back in StudyForge units."""
    if not math.isfinite(S_fsrs) or S_fsrs <= 0:
        raise ValueError(
            f"FSRS stability must be finite and greater than 0, got "
            f"{S_fsrs!r}"
        )
    return S_fsrs / _LN_10_OVER_9


class FSRSAdapter(ModelProtocol):
    """FSRS retrievability shape over the StudyForge review update."""

    version = "fsrs-adapter/1.0"

    def predict(self, state: ModelState, elapsed_days: float) -> float:
        if state.last_review is None:
            return 0.0
        elapsed = _check_elapsed(elapsed_days)
        p = (1.0 + FSRS_FACTOR * elapsed / state.S) ** (-FSRS_DECAY)
        return _check_probability(p)

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
                return state
            elapsed = day - state.last_review
        else:
            elapsed = 0
        if outcome is None:
            R = 0.0 if state.last_review is None else self.predict(
                state, elapsed)
        else:
            norm = normalize_outcome(outcome)
            if norm.kind != "binary":
                raise ValueError(
                    f"FSRS outcomes must be binary (0 or 1), got "
                    f"{outcome!r}; fractional recall is not an FSRS grade"
                )
            R = norm.value
            assert R is not None
            if R == 0.0:
                # An observed zero is a failed retrieval, not learning:
                # it changes neither stability nor the last review day.
                # The adapter reuses the StudyForge review update, so
                # it inherits the same rule.
                return state
        # Map through StudyForge units, apply the same update, map back.
        S_sf = to_studyforge_stability(state.S)
        from ..memory import GROWTH

        new_S_sf = S_sf * (1.0 + GROWTH * (1.0 - R))
        return ModelState(S=from_studyforge_stability(new_S_sf),
                          last_review=day, difficulty=state.difficulty)

    def fit(
        self, history: list[HistoryEvent], as_of: int | None = None,
        difficulty: int = 3,
    ) -> ModelState:
        events = _fit_events(history, as_of)
        for ev in events:
            if ev.outcome is not None and not (
                    ev.binary and ev.item is not None):
                raise ValueError(
                    "FSRS fits item-level binary outcomes only: event "
                    f"on day {ev.day} has no item id or is not marked "
                    "binary; use the exponential or HLR model for "
                    "topic-level fractional recall"
                )
        pairs = _usable_pairs(events)
        S_fit = fit_stability(pairs)
        if S_fit is None:
            from ..memory import initial_stability

            S_fit = initial_stability(difficulty)
        last = events[-1].day if events else None
        return ModelState(S=from_studyforge_stability(S_fit),
                          last_review=last, difficulty=difficulty)

    def from_stability(
        self, S: float, last_review: int | None, difficulty: int = 3
    ) -> ModelState:
        return ModelState(S=from_studyforge_stability(float(S)),
                          last_review=last_review, difficulty=difficulty)

    def to_stability(self, state: ModelState) -> float:
        return to_studyforge_stability(state.S)
