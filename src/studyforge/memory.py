"""Ebbinghaus forgetting curve and stability updates.

Memory model
------------
Each topic has a stability S measured in days. Recall after t days
without review is::

    R(t) = exp(-t / S)

Review update rule (SM-2 style, fully transparent)
--------------------------------------------------
On a review at elapsed time t with current stability S, first compute
retrievability R = exp(-t / S), then grow stability::

    S' = S * (1 + GROWTH * (1 - R))

with GROWTH = 1.0. Consequences of the rule, by construction:

* reviewing at full recall (R = 1) leaves S unchanged,
* reviewing after full decay (R = 0) doubles S,
* everything in between scales linearly with how much was forgotten.

The intuition: a successful recall after a long gap proves durable
memory, so stability grows more. A review right after learning proves
nothing new, so stability stays flat. Same-day repeat reviews gain
nothing, which stops the planner from cramming one topic.

Initial stability comes from a 1 to 5 difficulty rating::

    S0 = 3.0 / difficulty

so an easy topic (1) starts at 3.0 days and a hard one (5) at 0.6 days.

Personal stability fitting
--------------------------
The defaults above are a starting guess. When reviews are logged with an
observed recall fraction, each one becomes a data point: the days since
the previous review and what was actually remembered. Once a topic has
enough such points, ``fit_stability`` estimates its stability by least
squares on log recall and that fitted value replaces the default. The
fit is a running estimate: every new logged review refits it.
"""

from __future__ import annotations

import math

GROWTH = 1.0

# Personal stability fitting: least squares on log recall needs this many
# usable (interval, recall) pairs before the fitted value replaces the
# default. Fitted values are clamped to [S_MIN, S_MAX] days.
MIN_FIT_SAMPLES = 3
S_MIN = 0.1
S_MAX = 60.0


def fit_stability(pairs: list[tuple[float | None, float | None]]) -> float | None:
    """Fit a personal stability S from (interval days, recall) observations.

    Least squares on log recall: log R = -t / S. With u = 1/S, the
    minimizer of sum (log R_i + t_i * u)^2 is
    u = -sum(t_i * log R_i) / sum(t_i^2).

    Returns S clamped to [S_MIN, S_MAX], or None when fewer than
    MIN_FIT_SAMPLES usable pairs are given. Pairs with no interval, no
    recall, or a non-positive interval are skipped. Recall is clipped
    to [1e-3, 1.0], so a reported zero still counts as very fast
    forgetting instead of breaking the logarithm.
    """
    pts: list[tuple[float, float]] = []
    for t, R in pairs:
        if t is None or R is None:
            continue
        t = float(t)
        if t <= 0:
            continue
        Rc = min(1.0, max(1e-3, float(R)))
        pts.append((t, Rc))
    if len(pts) < MIN_FIT_SAMPLES:
        return None
    num = -sum(t * math.log(R) for t, R in pts)
    den = sum(t * t for t, _ in pts)
    u = num / den
    if u <= 0:
        # The user remembers everything: stability is effectively unbounded.
        return S_MAX
    return min(S_MAX, max(S_MIN, 1.0 / u))


def recall(t_days: float, S: float) -> float:
    """Probability of recall t days after the last review."""
    if S <= 0:
        return 0.0
    return math.exp(-max(0.0, t_days) / S)


def initial_stability(difficulty: int) -> float:
    """Stability in days for a fresh topic rated 1 (easy) to 5 (hard)."""
    d = min(5, max(1, int(difficulty)))
    return 3.0 / d


class TopicState:
    """Mutable memory state for one topic."""

    def __init__(
        self,
        name: str,
        difficulty: int = 3,
        S: float | None = None,
        last_review: int | None = None,
    ) -> None:
        self.name = name
        self.difficulty = difficulty
        self.S = initial_stability(difficulty) if S is None else float(S)
        self.last_review = last_review

    def recall_on(self, day: int) -> float:
        """Predicted recall on a given day offset."""
        if self.last_review is None:
            # Never studied: nothing learned, recall is zero.
            return 0.0
        return recall(day - self.last_review, self.S)

    def review(self, day: int, R: float | None = None) -> float:
        """Record a review on a day offset. Returns retrievability at review.

        Updates stability with the rule documented above and stamps the
        review day. If R is given, it is the observed recall fraction and
        is used for the update instead of the model's prediction. When
        the user reports what they actually remembered, the model
        believes them.
        """
        if R is None:
            R = self.recall_on(day)
        self.S = self.S * (1.0 + GROWTH * (1.0 - R))
        self.last_review = day
        return R

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "difficulty": self.difficulty,
            "S": self.S,
            "last_review": self.last_review,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TopicState":
        return cls(
            name=d["name"],
            difficulty=d.get("difficulty", 3),
            S=d.get("S"),
            last_review=d.get("last_review"),
        )
