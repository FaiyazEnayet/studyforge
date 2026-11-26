"""The naive constant baseline: predict the past mean recall.

This is the floor every fitted model must beat. It knows the base
rate and nothing else: one constant, the mean of the recorded
outcomes in the fitted history, predicted for every elapsed time.
With no recorded outcomes the constant is 0.5, and a topic never
reviewed still predicts 0.0, matching the other models.
"""

from __future__ import annotations

from .base import (
    HistoryEvent,
    ModelProtocol,
    ModelState,
    _check_day,
    _check_elapsed,
    _check_probability,
    normalize_outcome,
)
from .exponential import _fit_events


class ConstantModel(ModelProtocol):
    """One constant prediction: the mean of past recorded outcomes."""

    version = "constant/1.0"

    def __init__(self, constant: float | None = None) -> None:
        self.constant = constant

    @property
    def fitted_constant(self) -> float:
        """The fitted constant, or 0.5 before any outcomes are seen."""
        return 0.5 if self.constant is None else self.constant

    def predict(self, state: ModelState, elapsed_days: float) -> float:
        if state.last_review is None:
            # Never studied: nothing learned, recall is zero.
            return 0.0
        _check_elapsed(elapsed_days)
        return _check_probability(self.fitted_constant)

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
                # Same-day repeat review changes nothing.
                return state
        # The constant is fitted once per window, never refitted on
        # test outcomes; the transition only advances the last review.
        return ModelState(S=state.S, last_review=day,
                          difficulty=state.difficulty)

    def fit(
        self, history: list[HistoryEvent], as_of: int | None = None,
        difficulty: int = 3,
    ) -> ModelState:
        events = _fit_events(history, as_of)
        outcomes = [e.outcome for e in events if e.outcome is not None]
        for outcome in outcomes:
            normalize_outcome(outcome)
        self.constant = (
            sum(outcomes) / len(outcomes) if outcomes else None
        )
        last = events[-1].day if events else None
        # Stability is meaningless for a constant model; the state
        # carries a neutral 1.0 so the shared state type still holds.
        return ModelState(S=1.0, last_review=last, difficulty=difficulty)

    def to_dict(self, state: ModelState) -> dict:
        data = super().to_dict(state)
        data["constant"] = self.constant
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "ConstantModel":
        if not isinstance(data, dict) or data.get("model") != cls.version:
            raise ValueError(
                f"cannot load {cls.__name__} from data tagged "
                f"{data.get('model') if isinstance(data, dict) else data!r}; "
                f"expected {cls.version!r}"
            )
        return cls(constant=data.get("constant"))
