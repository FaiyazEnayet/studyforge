"""Pooled hierarchical half-life regression.

The half-life form ``R(t) = 2^(-t / h)`` is a reparameterization of
the exponential model (``h = S * ln 2``), so a per-topic half-life fit
alone would be the same regression in different units. The added
structure here is hierarchical pooling: each topic's rate is shrunk
toward a pooled rate estimated from every topic's history, which is
what helps sparse topics whose own few pairs would give a noisy or
default stability.

Fit, per topic, from usable (interval, recall) pairs:

* per-topic rate ``u_topic`` by least squares on log recall, exactly
  as the exponential fit (``u = 1 / S``);
* pooled rate ``u_pool`` the same way over all topics' pairs;
* shrunk rate ``u = (n * u_topic + k * u_pool) / (n + k)`` with
  ``n`` the topic's usable pair count and ``k = 4`` pseudo-observations
  of pooled strength.

With fewer than three usable pairs the topic has no trustworthy
per-topic estimate, so the pooled value stands (or the difficulty
default when nothing is pooled yet). Prediction and transition are
the exponential model's: the tournament therefore isolates the
effect of pooling, not of a different forgetting shape.
"""

from __future__ import annotations

import math

from ..memory import MIN_FIT_SAMPLES, S_MAX, S_MIN, initial_stability
from .base import HistoryEvent, ModelState, _usable_pairs
from .exponential import ExponentialModel, _fit_events

# Pseudo-observations of pooled strength in the shrinkage average.
SHRINKAGE_K = 4


def _rate_from_pairs(pairs: list[tuple[float, float]]) -> float | None:
    """Least-squares rate u = 1/S from (interval, recall) pairs.

    Returns None when the pairs carry no usable signal. Recall is
    clipped to [1e-3, 1.0], matching the exponential fit, so a
    reported zero still counts as very fast forgetting.
    """
    pts = [(t, min(1.0, max(1e-3, R))) for t, R in pairs if t > 0]
    if not pts:
        return None
    num = -sum(t * math.log(R) for t, R in pts)
    den = sum(t * t for t, _ in pts)
    if den <= 0:
        return None
    u = num / den
    if u <= 0:
        # The topic remembers everything: stability is unbounded.
        return 1.0 / S_MAX
    return min(1.0 / S_MIN, u)


class HierarchicalHalfLifeModel(ExponentialModel):
    """Exponential recall with pooled, shrunk stability fitting."""

    version = "hlr/1.0"

    def __init__(self, pooled_half_life: float | None = None) -> None:
        self.pooled_half_life = pooled_half_life

    @property
    def pooled_stability(self) -> float | None:
        """The pooled stability in StudyForge (e-folding) units."""
        if self.pooled_half_life is None:
            return None
        return self.pooled_half_life / math.log(2)

    @classmethod
    def fit_pooled(
        cls, histories: list[list[HistoryEvent]], as_of: int | None = None
    ) -> "HierarchicalHalfLifeModel":
        """Pooled half-life from every topic's history.

        Returns a model instance carrying the pooled value; per-topic
        fits on that instance shrink toward it. With no usable pairs
        anywhere the instance carries no pooled value and behaves like
        the exponential model.
        """
        pairs: list[tuple[float, float]] = []
        for history in histories:
            pairs.extend(_usable_pairs(_fit_events(history, as_of)))
        rate = _rate_from_pairs(pairs)
        if rate is None:
            return cls()
        return cls(pooled_half_life=math.log(2) / rate)

    def fit(
        self, history: list[HistoryEvent], as_of: int | None = None,
        difficulty: int = 3,
    ) -> ModelState:
        events = _fit_events(history, as_of)
        pairs = _usable_pairs(events)
        last = events[-1].day if events else None
        pooled_S = self.pooled_stability
        if len(pairs) < MIN_FIT_SAMPLES:
            # Too sparse for a per-topic estimate: pooled wins, else
            # the difficulty default, exactly as the baseline does.
            S = initial_stability(difficulty) if pooled_S is None else pooled_S
            return ModelState(S=S, last_review=last, difficulty=difficulty)
        u_topic = _rate_from_pairs(pairs)
        assert u_topic is not None
        if pooled_S is None:
            S = 1.0 / u_topic
        else:
            u_pool = 1.0 / pooled_S
            n = len(pairs)
            u = (n * u_topic + SHRINKAGE_K * u_pool) / (n + SHRINKAGE_K)
            S = 1.0 / u
        S = min(S_MAX, max(S_MIN, S))
        return ModelState(S=S, last_review=last, difficulty=difficulty)

    def to_dict(self, state: ModelState) -> dict:
        data = super().to_dict(state)
        data["pooled_half_life"] = self.pooled_half_life
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "HierarchicalHalfLifeModel":
        if not isinstance(data, dict) or data.get("model") != cls.version:
            raise ValueError(
                f"cannot load {cls.__name__} from data tagged "
                f"{data.get('model') if isinstance(data, dict) else data!r}; "
                f"expected {cls.version!r}"
            )
        return cls(pooled_half_life=data.get("pooled_half_life"))
