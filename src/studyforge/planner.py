"""Study planner: greedy allocation plus local search.

Problem
-------
Choose study sessions (topic, day) to maximize::

    sum over topics of weight * R(exam_day)

where R(exam_day) is predicted recall at exam time after simulating all
planned reviews through the memory model. Constraints: a daily minutes
cap, no sessions on rest days or on/after a topic's exam day, and at most
two sessions of one topic per day (a spacing guardrail against cramming).

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
    def __init__(
        self,
        name: str,
        weight: float,
        exam_day: int,
        S: float,
        last_review: int | None,
    ) -> None:
        self.name = name
        self.weight = float(weight)
        self.exam_day = int(exam_day)
        self.S = float(S)
        self.last_review = last_review


def simulate(topic: PlanTopic, session_days: list[int]) -> tuple[float, float]:
    """Simulate reviews on the given days. Returns (final S, recall at exam).

    A topic never reviewed starts with recall zero. Its first review is
    treated as full learning (R = 0 at review, so stability doubles),
    regardless of which day it falls on.
    """
    S = topic.S
    lr = topic.last_review
    for d in sorted(session_days):
        if lr is not None and (d <= lr or S <= 0):
            continue
        if lr is None:
            R = 0.0
        else:
            R = math.exp(-(d - lr) / S)
        S = S * (1.0 + GROWTH * (1.0 - R))
        lr = d
    if lr is None:
        return S, 0.0
    R_exam = math.exp(-(topic.exam_day - lr) / S) if S > 0 else 0.0
    return S, R_exam


def objective(topics: list[PlanTopic], plan: dict[str, list[int]]) -> float:
    """Sum of weight * predicted recall at exam time."""
    total = 0.0
    for tp in topics:
        _, R = simulate(tp, plan.get(tp.name, []))
        total += tp.weight * R
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
) -> dict[str, list[int]]:
    """Greedily add the highest marginal gain session until none helps."""
    plan: dict[str, list[int]] = {tp.name: [] for tp in topics}
    used: dict[int, int] = {}
    ordered = sorted(topics, key=lambda t: t.name)
    while True:
        base = objective(topics, plan)
        best: tuple[float, str, int] | None = None
        for tp in ordered:
            last_day = min(horizon, tp.exam_day)
            for d in range(start_day, last_day):
                if not _day_has_room(used, d, session_min, daily_cap, rest_days):
                    continue
                if plan[tp.name].count(d) >= MAX_PER_TOPIC_PER_DAY:
                    continue
                plan[tp.name].append(d)
                gain = objective(topics, plan) - base
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
        base = objective(topics, plan)
        used = used_minutes(plan)
        best: tuple[float, tuple] | None = None

        sessions = sorted((n, d) for n, ds in plan.items() for d in ds)
        for name, d in sessions:
            # candidate 1: drop the session
            plan[name].remove(d)
            gain = objective(topics, plan) - base
            plan[name].append(d)
            plan[name].sort()
            if gain > MIN_GAIN and (best is None or gain > best[0] + MIN_GAIN):
                best = (gain, ("drop", name, d))
            # candidate 2: move it to another (topic, day)
            for tp2 in ordered:
                last_day = min(horizon, tp2.exam_day)
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
                    gain = objective(topics, plan) - base
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
) -> tuple[dict[str, list[int]], float]:
    """Greedy allocation followed by local search.

    Returns (plan, objective value). Deterministic for a given seed;
    the seed is accepted for interface stability and the algorithm uses
    no randomness.
    """
    _ = seed
    rest = set(rest_days or ())
    plan = greedy(topics, horizon, daily_cap, session_min, rest, start_day)
    plan = local_search(topics, plan, horizon, daily_cap, session_min, rest, start_day)
    return plan, objective(topics, plan)


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
