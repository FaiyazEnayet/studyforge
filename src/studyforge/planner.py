"""Study planner: greedy allocation plus local search.

Problem
-------
Choose study sessions (topic, day) to maximize::

    sum over topics of weight * R(exam_day)

where R(exam_day) is predicted recall at exam time after simulating all
planned reviews through the memory model. Constraints: a daily minutes
cap, no sessions on rest days or on/after a topic's exam day, and at most
two sessions of one topic per day (a spacing guardrail against cramming).

Evaluation is as-of disciplined: each exam's recall is computed from
the memory state as of that exam's day (logged reviews are replayed,
never borrowed from a later day), and planning objectives take an
explicit as-of day so only exams still in the future participate in
forward replanning.

Algorithm
---------
1. Greedy: repeatedly add the single session with the largest marginal
   gain in the objective. Ties break by topic name then day, so the
   result is fully deterministic. The seed parameter is accepted for
   interface stability; the algorithm uses no randomness.
2. Local search: repeatedly apply the single best session move
   (relocate a session to another topic/day, or drop it) until no move
   improves the objective.

Sessions are fixed size chunks (default 25 minutes). The planner never
splits a chunk.
"""

from __future__ import annotations

import math

from .memory import GROWTH

MAX_PER_TOPIC_PER_DAY = 2
MIN_GAIN = 1e-12
MAX_PASSES = 100


class PlanTopic:
    """One knowledge topic with its exam bindings.

    A topic shared by several exams keeps one memory state (S,
    last_review) but contributes weight * recall at EACH bound exam's
    deadline to the objective. `bindings` is a list of
    (exam_name, weight, exam_day) tuples.

    `history` optionally carries the topic's logged review events as
    (day, observed_recall_or_None) pairs, and `difficulty` its 1-5
    rating. Together they let simulate rebuild the memory state as of
    an earlier exam day by event replay when the current state already
    includes reviews logged after that exam (see _asof_base). Topics
    built without them simply mark such an estimate unavailable.
    """

    def __init__(
        self,
        name: str,
        bindings: list[tuple[str, float, int]],
        S: float,
        last_review: int | None,
        difficulty: int | None = None,
        history: list[tuple[int, float | None]] | None = None,
    ) -> None:
        self.name = name
        self.bindings = [(str(e), float(w), int(d)) for e, w, d in bindings]
        self.S = float(S)
        self.last_review = last_review
        self.difficulty = difficulty
        self.history = (
            None if history is None
            else sorted((int(d), r) for d, r in history)
        )

    @property
    def latest_exam_day(self) -> int:
        return max(d for _, _, d in self.bindings)


def _asof_base(
    topic: PlanTopic, exam_day: int
) -> tuple[float, int | None] | None:
    """Memory state (S, last_review) as of `exam_day`, or None.

    Only logged reviews strictly before the exam day may inform an
    estimate for that day. If every logged review precedes the exam,
    the topic's current state already is the as-of state. Otherwise
    the state is rebuilt by replaying the topic's recorded review
    history from its initial stability, applying the memory model's
    update rule with the observed recall where one was logged.

    Returns None when the recorded history does not cover the current
    state (no history was supplied, or its latest event predates
    last_review, as in a migrated legacy state whose history names
    topics by display name): the as-of state is then unavailable, and
    the caller must mark the estimate unavailable instead of
    evaluating it from today's state.
    """
    lr = topic.last_review
    if lr is None or lr < exam_day:
        return topic.S, lr
    if topic.history is None or topic.difficulty is None:
        return None
    if not topic.history or max(d for d, _ in topic.history) < lr:
        return None
    from .memory import TopicState

    ts = TopicState(name=topic.name, difficulty=topic.difficulty)
    for d, r in topic.history:
        if d >= exam_day:
            break
        ts.review(d, R=r)
    return ts.S, ts.last_review


def simulate(
    topic: PlanTopic, session_days: list[int]
) -> tuple[float, list[float | None]]:
    """Simulate reviews on the given days.

    Returns (final S, recalls): recalls[i] is the predicted recall at
    bindings[i]'s exam day, using only sessions and logged reviews
    before that day. The logged state is taken as of the exam day
    (see _asof_base), never from a later day, so a shared topic
    evaluated at an earlier exam cannot see reviews logged for a
    later one. recalls[i] is None when that as-of state cannot be
    reconstructed from the recorded history: the estimate is
    unavailable, not fabricated. A topic never reviewed as of an exam
    day has recall zero there. Its first review is treated as full
    learning (R = 0 at review, so stability doubles), regardless of
    which day it falls on.
    """
    S, lr = topic.S, topic.last_review
    # chronological pass over planned sessions for the final S
    for d in sorted(session_days):
        if lr is not None and (d <= lr or S <= 0):
            continue
        if lr is None:
            R = 0.0
        else:
            R = math.exp(-(d - lr) / S)
        S = S * (1.0 + GROWTH * (1.0 - R))
        lr = d
    recalls: list[float | None] = []
    for _, _, exam_day in topic.bindings:
        base = _asof_base(topic, exam_day)
        if base is None:
            recalls.append(None)
            continue
        S_e, lr_e = base
        for d in sorted(session_days):
            if d >= exam_day:
                break
            if lr_e is not None and (d <= lr_e or S_e <= 0):
                continue
            if lr_e is None:
                R = 0.0
            else:
                R = math.exp(-(d - lr_e) / S_e)
            S_e = S_e * (1.0 + GROWTH * (1.0 - R))
            lr_e = d
        if lr_e is None or S_e <= 0:
            recalls.append(0.0)
        else:
            recalls.append(math.exp(-(exam_day - lr_e) / S_e))
    return S, recalls


def objective(
    topics: list[PlanTopic],
    plan: dict[str, list[int]],
    as_of: int | None = None,
) -> float:
    """Sum over topics and their exam bindings of weight * recall.

    With `as_of` given, only bindings whose exam day is after the
    as-of day contribute: a past exam's outcome can no longer be
    changed by planning, so forward replanning must not count (or
    chase) it. Bindings whose as-of recall is unavailable (None)
    never contribute.
    """
    total = 0.0
    for tp in topics:
        _, Rs = simulate(tp, plan.get(tp.name, []))
        for (_, w, exam_day), R in zip(tp.bindings, Rs):
            if R is None:
                continue
            if as_of is not None and exam_day <= as_of:
                continue
            total += w * R
    return total


def _day_has_room(
    used: dict[int, int],
    day: int,
    session_min: int,
    daily_cap: int,
    rest_days: set[int],
) -> bool:
    return day not in rest_days and used.get(day, 0) + session_min <= daily_cap


def greedy(
    topics: list[PlanTopic],
    horizon: int,
    daily_cap: int,
    session_min: int,
    rest_days: set[int],
    start_day: int = 0,
    as_of: int | None = None,
) -> dict[str, list[int]]:
    """Greedily add the highest marginal gain session until none helps."""
    plan: dict[str, list[int]] = {tp.name: [] for tp in topics}
    used: dict[int, int] = {}
    ordered = sorted(topics, key=lambda t: t.name)
    while True:
        base = objective(topics, plan, as_of=as_of)
        best: tuple[float, str, int] | None = None
        for tp in ordered:
            last_day = min(horizon, tp.latest_exam_day)
            for d in range(start_day, last_day):
                if not _day_has_room(used, d, session_min, daily_cap, rest_days):
                    continue
                if plan[tp.name].count(d) >= MAX_PER_TOPIC_PER_DAY:
                    continue
                plan[tp.name].append(d)
                gain = objective(topics, plan, as_of=as_of) - base
                plan[tp.name].pop()
                if gain > MIN_GAIN and (best is None or gain > best[0] + MIN_GAIN):
                    best = (gain, tp.name, d)
        if best is None:
            break
        _, name, d = best
        plan[name].append(d)
        plan[name].sort()
        used[d] = used.get(d, 0) + session_min
    return plan


def local_search(
    topics: list[PlanTopic],
    plan: dict[str, list[int]],
    horizon: int,
    daily_cap: int,
    session_min: int,
    rest_days: set[int],
    start_day: int = 0,
    max_passes: int = MAX_PASSES,
    caps: dict[int, int] | None = None,
    as_of: int | None = None,
) -> dict[str, list[int]]:
    """Best improvement moves until convergence. Deterministic."""
    plan = {k: sorted(v) for k, v in plan.items()}
    ordered = sorted(topics, key=lambda t: t.name)
    by_name = {tp.name: tp for tp in topics}

    def cap_of(day: int) -> int:
        return caps.get(day, daily_cap) if caps else daily_cap

    def used_minutes(p: dict[str, list[int]]) -> dict[int, int]:
        u: dict[int, int] = {}
        for days in p.values():
            for d in days:
                u[d] = u.get(d, 0) + session_min
        return u

    for _ in range(max_passes):
        base = objective(topics, plan, as_of=as_of)
        used = used_minutes(plan)
        best: tuple[float, tuple] | None = None

        sessions = sorted((n, d) for n, ds in plan.items() for d in ds)
        for name, d in sessions:
            # candidate 1: drop the session
            plan[name].remove(d)
            gain = objective(topics, plan, as_of=as_of) - base
            plan[name].append(d)
            plan[name].sort()
            if gain > MIN_GAIN and (best is None or gain > best[0] + MIN_GAIN):
                best = (gain, ("drop", name, d))
            # candidate 2: move it to another (topic, day)
            for tp2 in ordered:
                last_day = min(horizon, tp2.latest_exam_day)
                for d2 in range(start_day, last_day):
                    if (tp2.name, d2) == (name, d):
                        continue
                    if d2 in rest_days:
                        continue
                    room = used.get(d2, 0) + (0 if d2 == d else session_min)
                    if room > cap_of(d2):
                        continue
                    if tp2.name == name:
                        cnt = plan[name].count(d2)
                    else:
                        cnt = plan[tp2.name].count(d2)
                    if cnt >= MAX_PER_TOPIC_PER_DAY:
                        continue
                    plan[name].remove(d)
                    plan[tp2.name].append(d2)
                    plan[tp2.name].sort()
                    gain = objective(topics, plan, as_of=as_of) - base
                    plan[tp2.name].remove(d2)
                    plan[name].append(d)
                    plan[name].sort()
                    if gain > MIN_GAIN and (best is None or gain > best[0] + MIN_GAIN):
                        best = (gain, ("move", name, d, tp2.name, d2))
        if best is None:
            break
        _, action = best
        if action[0] == "drop":
            _, name, d = action
            plan[name].remove(d)
        else:
            _, name, d, name2, d2 = action
            plan[name].remove(d)
            plan[name2].append(d2)
            plan[name2].sort()
    return plan


def build_plan(
    topics: list[PlanTopic],
    horizon: int,
    daily_cap: int,
    session_min: int,
    rest_days: set[int] | None = None,
    start_day: int = 0,
    seed: int = 0,
    as_of: int | None = None,
) -> tuple[dict[str, list[int]], float]:
    """Greedy allocation followed by local search.

    Returns (plan, objective value). Deterministic for a given seed;
    the seed is accepted for interface stability and the algorithm uses
    no randomness.

    The objective is evaluated as of `as_of` (default: `start_day`,
    the day planning starts): only exams still in the future at that
    day participate, so replanning never counts or chases a past
    exam's contribution.
    """
    _ = seed
    rest = set(rest_days or ())
    if as_of is None:
        as_of = start_day
    plan = greedy(topics, horizon, daily_cap, session_min, rest, start_day,
                  as_of=as_of)
    plan = local_search(topics, plan, horizon, daily_cap, session_min, rest,
                        start_day, as_of=as_of)
    return plan, objective(topics, plan, as_of=as_of)


def to_schedule(
    plan: dict[str, list[int]], session_min: int
) -> dict[int, list[tuple[str, int]]]:
    """Convert topic -> days into day -> [(topic, minutes)]."""
    sched: dict[int, list[tuple[str, int]]] = {}
    for name in sorted(plan):
        for d in sorted(plan[name]):
            sched.setdefault(d, []).append((name, session_min))
    for d in sched:
        sched[d].sort()
    return sched
