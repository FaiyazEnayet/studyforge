"""Explicit learning activities: what the student actually did.

A generic 25-minute review used to be modeled as an automatic
stability increase even if recall failed or the task was first
learning, rereading, retrieval, problem solving, or corrective
feedback. This module records what an activity was, and applies
only the transition effect that activity can support.

Activity kinds
--------------
* ``learn``: first exposure to the material (acquisition).
* ``retrieval``: an attempt to recall it (a test, a flashcard).
* ``worked-example``: studying a worked solution (acquisition).
* ``practice-problem``: solving a problem, on a repeated or a novel
  item (novel items are transfer, repeated items are retrieval).
* ``explain``: explaining the topic in your own words (retrieval).
* ``feedback``: corrective feedback after an attempt. Feedback
  alone is never learning: it is recorded, and its effect is left
  for a fitted model to find once enough data exists.

Labels
------
Every activity carries one label: ``acquisition`` (learn and
worked-example), ``retrieval`` (retrieval, explain, and
practice-problem on a repeated item), ``transfer`` (a
retrieval-family activity on a novel item), or ``feedback``.
Transfer is claimed only when the item is marked novel; a novel
item is never assumed.

Transition rules (the observational stage)
------------------------------------------
Only what was observed moves the memory state:

* acquisition on a fresh topic establishes it: the last review day
  is stamped and stability stays at its initial value. Later
  acquisition (rereading, another worked example) is recorded and
  claims nothing.
* a successful retrieval (observed correctness above 0) grows
  stability by the review update, using the observed correctness,
  and stamps the review day.
* a failed retrieval (observed correctness 0) changes nothing:
  stability and the last review day stay put. The attempt stays in
  the log for fitting, but the memory state does not credit it.
* an activity with no recorded correctness has an unknown outcome:
  it is recorded and changes nothing. Missing outcomes are never
  treated as 0.
* feedback alone changes nothing.
* a same-day repeat is recorded and changes nothing, matching the
  planner and the model transition.

These are the rules for recording what happened. Separate
transition effects per activity kind are a fitting question for
later, once enough logged data exists; this module does not claim
them. The ablation in this module measures whether the structure
above predicts held-out logged outcomes better than the old update,
which credited every logged event including failures.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .memory import GROWTH, TopicState, initial_stability, recall

ACTIVITY_KINDS = (
    "learn",
    "retrieval",
    "worked-example",
    "practice-problem",
    "explain",
    "feedback",
)

ITEM_NOVELTY = ("repeated", "novel")

# Outcome vocabulary for apply_activity results.
OUTCOMES = ("established", "success", "failure", "unknown", "recorded")

_MIN_EVENTS_FOR_ABLATION = 5


def _check_day(day: int) -> int:
    if isinstance(day, bool) or not isinstance(day, int):
        raise ValueError(
            f"activity day must be a whole number of days, got {day!r}"
        )
    if day < 0:
        raise ValueError(
            f"activity day must be 0 or later, got {day}"
        )
    return day


def _check_minutes(minutes: int) -> int:
    if isinstance(minutes, bool) or not isinstance(minutes, int):
        raise ValueError(
            f"activity minutes must be a whole number, got {minutes!r}"
        )
    if minutes <= 0:
        raise ValueError(
            f"activity minutes must be greater than 0, got {minutes}"
        )
    return minutes


def _check_correctness(correctness) -> float | None:
    if correctness is None:
        return None
    if isinstance(correctness, bool) or not isinstance(correctness, (int, float)):
        raise ValueError(
            f"correctness must be a number between 0 and 1, got "
            f"{correctness!r}; log the fraction actually remembered, "
            "or 0/1 for a right-or-wrong retrieval"
        )
    f = float(correctness)
    if not math.isfinite(f) or not 0.0 <= f <= 1.0:
        raise ValueError(
            f"correctness must be between 0 and 1, got {correctness!r}"
        )
    return f


@dataclass(frozen=True)
class Activity:
    """One logged learning activity.

    ``kind`` is one of ACTIVITY_KINDS. ``day`` is the day offset and
    ``minutes`` the session length. ``correctness`` is the observed
    outcome (fraction remembered, or 0/1 for a right-or-wrong
    retrieval), or None when nothing was recorded. ``item`` marks a
    retrieval-family activity as ``repeated`` or ``novel`` (a novel
    item is a transfer test); None means the item was not marked.
    ``assistance`` records whether help was given (a hint, notes, a
    peek at the solution) and ``feedback`` whether corrective
    feedback followed. Assistance and feedback are recorded only;
    neither claims a transition effect yet.
    """

    kind: str
    day: int
    minutes: int
    correctness: float | None = None
    item: str | None = None
    assistance: bool = False
    feedback: bool = False

    def __post_init__(self) -> None:
        if self.kind not in ACTIVITY_KINDS:
            raise ValueError(
                f"unknown activity kind {self.kind!r}; choose one of "
                f"{list(ACTIVITY_KINDS)}"
            )
        object.__setattr__(self, "day", _check_day(self.day))
        object.__setattr__(self, "minutes", _check_minutes(self.minutes))
        object.__setattr__(
            self, "correctness", _check_correctness(self.correctness))
        if self.item is not None and self.item not in ITEM_NOVELTY:
            raise ValueError(
                f"item novelty must be one of {list(ITEM_NOVELTY)}, got "
                f"{self.item!r}; mark a new problem 'novel' and a "
                "repeat 'repeated'"
            )
        if not isinstance(self.assistance, bool):
            raise ValueError(
                f"assistance must be true or false, got "
                f"{self.assistance!r}"
            )
        if not isinstance(self.feedback, bool):
            raise ValueError(
                f"feedback must be true or false, got {self.feedback!r}"
            )


def label(activity: Activity) -> str:
    """The learning label for an activity.

    Acquisition, retrieval, transfer, and feedback never conflate:
    transfer needs a novel item on a retrieval-family activity, and
    feedback is its own label.
    """
    if activity.kind in ("learn", "worked-example"):
        return "acquisition"
    if activity.kind == "feedback":
        return "feedback"
    if activity.item == "novel":
        return "transfer"
    return "retrieval"


@dataclass(frozen=True)
class ActivityResult:
    """What applying one activity did to the memory state."""

    activity: Activity
    label: str
    outcome: str  # one of OUTCOMES
    S_before: float
    S_after: float
    last_review_before: int | None
    last_review_after: int | None


def apply_activity(ts: TopicState, activity: Activity) -> ActivityResult:
    """Apply one logged activity to a topic's memory state.

    Returns what happened: the label, the outcome, and the state
    before and after. See the module docstring for the rules. An
    activity before the last review day raises ValueError; history
    runs forward in time.
    """
    S_before, lr_before = ts.S, ts.last_review
    if lr_before is not None and activity.day < lr_before:
        raise ValueError(
            f"activity on day {activity.day} comes before the last "
            f"review on day {lr_before}; history must run forward "
            "in time"
        )

    def done(outcome: str) -> ActivityResult:
        return ActivityResult(
            activity=activity,
            label=label(activity),
            outcome=outcome,
            S_before=S_before,
            S_after=ts.S,
            last_review_before=lr_before,
            last_review_after=ts.last_review,
        )

    if lr_before is not None and activity.day == lr_before:
        # A same-day repeat proves nothing new.
        return done("recorded")
    if activity.kind in ("learn", "worked-example"):
        if lr_before is None:
            # First exposure establishes the topic at its initial
            # stability; acquisition is not a doubling.
            ts.last_review = activity.day
            return done("established")
        return done("recorded")
    if activity.kind == "feedback":
        return done("recorded")
    # Retrieval family: retrieval, explain, practice-problem.
    if activity.correctness is None:
        return done("unknown")
    if activity.correctness == 0.0:
        return done("failure")
    ts.review(activity.day, R=float(activity.correctness))
    return done("success")


def replay_activities(
    name: str,
    difficulty: int,
    activities: list[Activity],
    S: float | None = None,
) -> TopicState:
    """Rebuild a topic's memory state from its logged activities.

    Activities replay in day order (stable, so the input order of
    same-day events is kept). Replay is deterministic: the same
    activities always rebuild the same state.
    """
    ts = TopicState(name, difficulty=difficulty, S=S)
    for act in sorted(activities, key=lambda a: a.day):
        apply_activity(ts, act)
    return ts


def _baseline_transition(
    S: float, last_review: int | None, day: int,
    correctness: float | None,
) -> tuple[float, int | None]:
    """One step of the pre-A07 update, kept only for the ablation.

    The old rule treated every logged event as a review: an observed
    zero doubled stability, a missing outcome used the model's
    prediction, and a first event doubled stability from the
    initial value. Same-day repeats change nothing.
    """
    if last_review is not None:
        if day < last_review:
            raise ValueError(
                f"activity on day {day} comes before the last review "
                f"on day {last_review}; history must run forward "
                "in time"
            )
        if day == last_review:
            return S, last_review
        R = recall(day - last_review, S) if correctness is None \
            else float(correctness)
    else:
        R = 0.0 if correctness is None else float(correctness)
    return S * (1.0 + GROWTH * (1.0 - R)), day


def _replay_baseline(
    difficulty: int, activities: list[Activity],
) -> tuple[float, int | None]:
    """(S, last_review) after replaying events with the old update."""
    S = initial_stability(difficulty)
    last_review: int | None = None
    for act in sorted(activities, key=lambda a: a.day):
        S, last_review = _baseline_transition(
            S, last_review, act.day, act.correctness)
    return S, last_review


def _insufficient(n_events: int, n_train: int, n_holdout: int,
                  n_scored: int) -> dict:
    """An ablation result that reports too little data, honestly."""
    return {
        "n_events": n_events,
        "n_train": n_train,
        "n_holdout": n_holdout,
        "n_scored": n_scored,
        "baseline_mae": None,
        "activity_mae": None,
        "difference": None,
        "verdict": "insufficient data",
    }


def ablation(activities: list[Activity], difficulty: int = 3) -> dict:
    """Does the activity structure predict held-out outcomes?

    The events split chronologically: the last fifth is the holdout,
    the rest is the train set. Both rules replay the train set, then
    score one-step-ahead on the holdout in day order: each held-out
    event is predicted from the state as of its day, scored against
    its observed correctness, and then applied under the same rule
    before the next prediction. Held-out events with no recorded
    outcome are skipped, never scored as 0. The split and the
    scoring are deterministic.

    Returns the counts, the mean absolute error of each rule, their
    difference (baseline minus activity-aware, so a positive number
    favors the activity-aware rule), and a verdict: ``activity-aware``,
    ``baseline``, ``tie``, or ``insufficient data`` when there are
    too few events or nothing scorable in the holdout.
    """
    events = sorted(activities, key=lambda a: a.day)
    n = len(events)
    n_holdout = max(1, n // 5)
    n_train = n - n_holdout
    if n < _MIN_EVENTS_FOR_ABLATION or n_train < 4:
        return _insufficient(n, n_train, n_holdout, 0)
    train, holdout = events[:n_train], events[n_train:]

    # Baseline rollout: the old update.
    base_S, base_lr = _replay_baseline(difficulty, train)
    base_errors: list[float] = []
    for act in holdout:
        pred = 0.0 if base_lr is None else recall(act.day - base_lr, base_S)
        if act.correctness is not None:
            base_errors.append(abs(pred - float(act.correctness)))
        base_S, base_lr = _baseline_transition(
            base_S, base_lr, act.day, act.correctness)

    # Activity-aware rollout.
    aware = replay_activities("ablation", difficulty, train)
    aware_errors: list[float] = []
    for act in holdout:
        pred = aware.recall_on(act.day)
        if act.correctness is not None:
            aware_errors.append(abs(pred - float(act.correctness)))
        apply_activity(aware, act)

    n_scored = len(base_errors)
    if n_scored == 0:
        return _insufficient(n, n_train, n_holdout, 0)
    baseline_mae = sum(base_errors) / n_scored
    activity_mae = sum(aware_errors) / n_scored
    difference = baseline_mae - activity_mae
    if difference > 1e-12:
        verdict = "activity-aware"
    elif difference < -1e-12:
        verdict = "baseline"
    else:
        verdict = "tie"
    return {
        "n_events": n,
        "n_train": n_train,
        "n_holdout": n_holdout,
        "n_scored": n_scored,
        "baseline_mae": baseline_mae,
        "activity_mae": activity_mae,
        "difference": difference,
        "verdict": verdict,
    }
