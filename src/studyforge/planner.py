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
   improves the objective. After each accepted relocation, the freed
   capacity is refilled by deterministic re-augmentation (best
   positive-gain sessions first).
3. Exchange search: when the oracle gap justifies it, bounded
   two-session swaps (two sessions exchange days) are evaluated and
   the single best improving swap is applied, followed by a settle
   pass of local search plus re-augmentation.

Speed comes from delta scoring, not from changing what is optimized.
A session added to, dropped from, or moved between topics changes
only the involved topics' contributions, so every candidate gain is
computed from those topics alone. Per-topic trajectories are cached
as chronological snapshots (the state after each effective review, in
day order) keyed by the full session list, so repeated evaluations of
the same plan reuse the same snapshots instead of replaying history.
The cache is owned by one search run and invalidated explicitly;
``objective`` itself still recomputes from scratch and stays the
independent check that delta totals match full recomputation.

Sessions are fixed size chunks (default 25 minutes). The planner never
splits a chunk.
"""

from __future__ import annotations

import math

from .models import ExponentialModel, get_model

# The model used whenever none is chosen: the unchanged baseline.
_DEFAULT_MODEL = ExponentialModel()

MAX_PER_TOPIC_PER_DAY = 2
MIN_GAIN = 1e-12
MAX_PASSES = 100
# A two-session exchange phase runs only when the oracle reference
# says meaningful headroom remains: the relative gap (oracle minus
# achieved, over the oracle) must clear this threshold.
EXCHANGE_GAP_THRESHOLD = 0.01
MAX_EXCHANGE_ROUNDS = 5
MAX_EXCHANGE_EVALS = 5000
MAX_SETTLE_ROUNDS = 3


def _resolve_model(model, topic=None):
    """The model for a computation: explicit, then the topic's, then default."""
    if model is not None:
        return model
    if topic is not None and getattr(topic, "model", None) is not None:
        return topic.model
    return _DEFAULT_MODEL


def _require_int(value, label: str) -> int:
    """A whole-number input, or a ValueError naming the fix."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"{label} must be a whole number, got {value!r}; "
            "use a plain number without quotes or a decimal point"
        )
    return value


def _require_positive_int(value, label: str) -> int:
    """A whole number greater than zero, or a ValueError naming the fix."""
    v = _require_int(value, label)
    if v <= 0:
        raise ValueError(
            f"{label} must be greater than 0, got {v}; "
            "use a positive number"
        )
    return v


def _require_finite(value, label: str) -> float:
    """A finite number, or a ValueError naming the fix."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"{label} must be a number, got {value!r}; "
            "use a plain number without quotes"
        )
    f = float(value)
    if not math.isfinite(f):
        raise ValueError(
            f"{label} must be finite, got {value!r}; "
            "replace it with an ordinary number"
        )
    return f


def _validate_schedule_args(topics, horizon, daily_cap, session_min,
                            rest_days, start_day, as_of=None, caps=None,
                            max_passes=None):
    """Check the scheduling parameters shared by greedy and local search.

    Zero or negative session lengths and daily caps, a negative horizon
    or start day, and non-positive per-day caps are rejected with an
    explicit error naming the bad value.
    """
    if not isinstance(topics, (list, tuple)) or not topics:
        raise ValueError(
            "build a plan for at least one topic: the topic list is empty"
        )
    horizon = _require_int(horizon, "horizon")
    if horizon < 0:
        raise ValueError(
            f"horizon must be 0 or later, got {horizon}; use the latest "
            "exam day as the horizon"
        )
    _require_positive_int(daily_cap, "daily_cap")
    _require_positive_int(session_min, "session_min")
    start_day = _require_int(start_day, "start_day")
    if start_day < 0:
        raise ValueError(
            f"start_day must be 0 or later, got {start_day}; days count "
            "forward from today (day 0)"
        )
    if rest_days is not None:
        for d in rest_days:
            dv = _require_int(d, "rest day")
            if dv < 0:
                raise ValueError(
                    f"rest day must be 0 or later, got {dv}"
                )
    if as_of is not None:
        as_of = _require_int(as_of, "as_of")
        if as_of < 0:
            raise ValueError(
                f"as_of must be 0 or later, got {as_of}"
            )
    if caps is not None:
        for day, cap in caps.items():
            dv = _require_int(day, "cap day")
            if dv < 0:
                raise ValueError(f"cap day must be 0 or later, got {dv}")
            _require_positive_int(cap, f"cap for day {dv}")
    if max_passes is not None:
        _require_positive_int(max_passes, "max_passes")


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
        model=None,
    ) -> None:
        if not bindings:
            raise ValueError(
                f"topic {name!r} needs at least one exam binding: bind "
                "it to an exam with a weight and an exam day"
            )
        seen_exams: set[str] = set()
        for e, w, d in bindings:
            if not isinstance(e, str) or not e.strip():
                raise ValueError(
                    f"each binding for topic {name!r} needs a non-empty "
                    "exam name"
                )
            if e in seen_exams:
                raise ValueError(
                    f"duplicate binding for exam {e!r} on topic {name!r}: "
                    "bind the topic to each exam once"
                )
            seen_exams.add(e)
            wv = _require_finite(
                w, f"weight for topic {name!r} in exam {e!r}")
            if wv <= 0:
                raise ValueError(
                    f"weight for topic {name!r} in exam {e!r} must be "
                    f"greater than 0, got {wv}"
                )
            dv = _require_int(
                d, f"exam day for topic {name!r} in exam {e!r}")
            if dv < 0:
                raise ValueError(
                    f"exam day for topic {name!r} in exam {e!r} must be "
                    f"0 or later, got {dv}"
                )
        s_val = _require_finite(S, f"stability for topic {name!r}")
        if s_val <= 0:
            raise ValueError(
                f"stability for topic {name!r} must be greater than 0, "
                f"got {s_val}; stability comes from the difficulty "
                "rating or a fitted value"
            )
        if last_review is not None:
            lr = _require_int(last_review, f"last_review for topic {name!r}")
            if lr < 0:
                raise ValueError(
                    f"last_review for topic {name!r} must be 0 or later, "
                    f"got {lr}"
                )
        if difficulty is not None:
            diff = _require_int(difficulty, f"difficulty for topic {name!r}")
            if not 1 <= diff <= 5:
                raise ValueError(
                    f"difficulty for topic {name!r} must be between 1 "
                    f"and 5, got {diff}"
                )
        if history is not None:
            for d, r in history:
                dv = _require_int(d, f"review day for topic {name!r}")
                if dv < 0:
                    raise ValueError(
                        f"review day for topic {name!r} must be 0 or "
                        f"later, got {dv}"
                    )
                if r is not None:
                    rv = _require_finite(r, f"recall for topic {name!r}")
                    if not 0.0 <= rv <= 1.0:
                        raise ValueError(
                            f"recall for topic {name!r} must be between 0 "
                            f"and 1, got {rv}; log the fraction actually "
                            "remembered"
                        )
        self.name = name
        self.bindings = [(str(e), float(w), int(d)) for e, w, d in bindings]
        self.S = float(S)
        self.last_review = last_review
        self.difficulty = difficulty
        self.history = (
            None if history is None
            else sorted((int(d), r) for d, r in history)
        )
        # The memory model this topic is evaluated with. None means
        # the default exponential model, so existing callers change
        # nothing by leaving it out.
        self.model = model

    @property
    def latest_exam_day(self) -> int:
        return max(d for _, _, d in self.bindings)


def _asof_base(
    topic: PlanTopic, exam_day: int, model=None
) -> tuple[float, int | None] | None:
    """Memory state (S, last_review) as of `exam_day`, or None.

    Only logged reviews strictly before the exam day may inform an
    estimate for that day. If every logged review precedes the exam,
    the topic's current state already is the as-of state. Otherwise
    the state is rebuilt by replaying the topic's recorded review
    history from its initial stability through the model, applying
    the observed recall where one was logged. The returned stability
    is in StudyForge units whatever model ran the replay.

    Returns None when the recorded history does not cover the current
    state (no history was supplied, or its latest event predates
    last_review, as in a migrated legacy state whose history names
    topics by display name): the as-of state is then unavailable, and
    the caller must mark the estimate unavailable instead of
    evaluating it from today's state.
    """
    model = _resolve_model(model, topic)
    lr = topic.last_review
    if lr is None or lr < exam_day:
        return topic.S, lr
    if topic.history is None or topic.difficulty is None:
        return None
    if not topic.history or max(d for d, _ in topic.history) < lr:
        return None
    from .memory import initial_stability

    state = model.from_stability(
        initial_stability(topic.difficulty), None, topic.difficulty)
    for d, r in topic.history:
        if d >= exam_day:
            break
        state = model.transition(state, d, outcome=r)
    return model.to_stability(state), state.last_review


def _replay(model, base_state, days, upto=None):
    """Chronological replay of planned sessions from ``base_state``.

    Returns the snapshot list: ``(day, state)`` after each effective
    transition, in day order. A session on or before the last review
    day is a no-op and leaves no snapshot; replay stops at ``upto``.
    The states are the model's own immutable states, so a snapshot
    stays valid for as long as its inputs do. The skip rule and the
    transition order are exactly the ones ``simulate`` has always
    used, so replaying from a snapshot gives the same state as
    replaying from the base.
    """
    snaps = []
    state = base_state
    S, lr = model.to_stability(state), state.last_review
    for d in days:
        if upto is not None and d >= upto:
            break
        if lr is not None and (d <= lr or S <= 0):
            continue
        state = model.transition(state, d)
        S, lr = model.to_stability(state), state.last_review
        snaps.append((d, state))
    return snaps


def _state_at(snaps, base_state, day):
    """The state after all sessions before ``day``.

    The last snapshot strictly before ``day``, or the base state when
    no session precedes it.
    """
    state = base_state
    for d, s in snaps:
        if d >= day:
            break
        state = s
    return state


def simulate(
    topic: PlanTopic, session_days: list[int], model=None
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
    which day it falls on. The final S is in StudyForge units.

    `model` selects the memory model (a model instance, or a registry
    name); with none given the topic's model applies, else the
    default exponential model, so existing callers see the unchanged
    baseline behavior.
    """
    if isinstance(model, str):
        model = get_model(model)
    model = _resolve_model(model, topic)
    S, lr = topic.S, topic.last_review
    for d in session_days:
        dv = _require_int(d, f"session day for topic {topic.name!r}")
        if dv < 0:
            raise ValueError(
                f"session day for topic {topic.name!r} must be 0 or "
                f"later, got {dv}"
            )
    days = sorted(session_days)
    difficulty = topic.difficulty or 3
    # chronological pass over planned sessions for the final S
    base_state = model.from_stability(S, lr, difficulty)
    snaps = _replay(model, base_state, days)
    if snaps:
        S = model.to_stability(snaps[-1][1])
    recalls: list[float | None] = []
    for _, _, exam_day in topic.bindings:
        base = _asof_base(topic, exam_day, model)
        if base is None:
            recalls.append(None)
            continue
        S_e, lr_e = base
        state_e = model.from_stability(S_e, lr_e, difficulty)
        snaps_e = _replay(model, state_e, days, upto=exam_day)
        if snaps_e:
            state_e = snaps_e[-1][1]
            S_e, lr_e = model.to_stability(state_e), state_e.last_review
        if lr_e is None or S_e <= 0:
            recalls.append(0.0)
        else:
            recalls.append(model.predict(state_e, exam_day - lr_e))
    return S, recalls


def objective(
    topics: list[PlanTopic],
    plan: dict[str, list[int]],
    as_of: int | None = None,
    model=None,
) -> float:
    """Sum over topics and their exam bindings of weight * recall.

    With `as_of` given, only bindings whose exam day is after the
    as-of day contribute: a past exam's outcome can no longer be
    changed by planning, so forward replanning must not count (or
    chase) it. Bindings whose as-of recall is unavailable (None)
    never contribute. `model` selects the memory model; with none
    given each topic's own model applies, else the default.
    """
    if as_of is not None:
        as_of = _require_int(as_of, "as_of")
        if as_of < 0:
            raise ValueError(f"as_of must be 0 or later, got {as_of}")
    total = 0.0
    for tp in topics:
        _, Rs = simulate(tp, plan.get(tp.name, []), model=model)
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


class _ObjectiveCache:
    """Per-topic trajectory cache for one search run.

    Three memo tables, all keyed by values that fully determine the
    result, so a cached entry can never go stale while its inputs
    live: the as-of base per (topic, exam day, model), the
    chronological snapshot list per (topic, base state, session
    list, model), and the whole simulate result per (topic, session
    list, model). Topics are fixed for the run, so the topic name
    identifies the topic. ``invalidate`` drops a topic's entries (or
    all of them) explicitly; it exists so a caller that replaces a
    topic's data can force a recompute instead of trusting the key.
    ``hits`` and ``misses`` count simulate-level lookups.
    """

    def __init__(self):
        self._asof: dict = {}
        self._trajectories: dict = {}
        self._simulate: dict = {}
        self.hits = 0
        self.misses = 0

    @property
    def entries(self) -> int:
        return len(self._asof) + len(self._trajectories) + len(self._simulate)

    def invalidate(self, name: str | None = None) -> None:
        """Drop cached entries for one topic, or all of them."""
        if name is None:
            self._asof.clear()
            self._trajectories.clear()
            self._simulate.clear()
            return
        for table in (self._asof, self._trajectories, self._simulate):
            for key in [k for k in table if k[0] == name]:
                del table[key]

    def asof_base(self, topic: PlanTopic, exam_day: int, model):
        key = (topic.name, exam_day, id(model))
        if key not in self._asof:
            self._asof[key] = _asof_base(topic, exam_day, model)
        return self._asof[key]

    def trajectory(self, topic: PlanTopic, base_state, days, model):
        """Chronological snapshots from ``base_state`` over ``days``."""
        key = (topic.name, base_state, tuple(days), id(model))
        snaps = self._trajectories.get(key)
        if snaps is None:
            snaps = _replay(model, base_state, list(days))
            self._trajectories[key] = snaps
        return snaps

    def simulate(self, topic: PlanTopic, days, model=None):
        """The topic's simulate result, memoized for this run."""
        if isinstance(model, str):
            model = get_model(model)
        model = _resolve_model(model, topic)
        key = (topic.name, tuple(sorted(days)), id(model))
        hit = self._simulate.get(key)
        if hit is not None:
            self.hits += 1
            return hit[0], list(hit[1])
        self.misses += 1
        for d in days:
            dv = _require_int(d, f"session day for topic {topic.name!r}")
            if dv < 0:
                raise ValueError(
                    f"session day for topic {topic.name!r} must be 0 or "
                    f"later, got {dv}"
                )
        days_sorted = sorted(days)
        difficulty = topic.difficulty or 3
        base_state = model.from_stability(
            topic.S, topic.last_review, difficulty)
        snaps = self.trajectory(topic, base_state, days_sorted, model)
        S = model.to_stability(snaps[-1][1]) if snaps else topic.S
        recalls: list[float | None] = []
        for _, _, exam_day in topic.bindings:
            base = self.asof_base(topic, exam_day, model)
            if base is None:
                recalls.append(None)
                continue
            S_e, lr_e = base
            state_e = model.from_stability(S_e, lr_e, difficulty)
            snaps_e = self.trajectory(topic, state_e, days_sorted, model)
            state_e = _state_at(snaps_e, state_e, exam_day)
            S_e, lr_e = model.to_stability(state_e), state_e.last_review
            if lr_e is None or S_e <= 0:
                recalls.append(0.0)
            else:
                recalls.append(model.predict(state_e, exam_day - lr_e))
        result = (S, tuple(recalls))
        self._simulate[key] = result
        return result[0], list(result[1])

    def contribution(self, topic: PlanTopic, days, as_of=None, model=None):
        """The topic's share of the objective for ``days``."""
        _, recalls = self.simulate(topic, days, model)
        total = 0.0
        for (_, w, exam_day), R in zip(topic.bindings, recalls):
            if R is None:
                continue
            if as_of is not None and exam_day <= as_of:
                continue
            total += w * R
        return total


class _SearchRun:
    """Delta-evaluated search state for one greedy/local-search run.

    Holds the plan, each topic's current contribution (kept current
    as the plan changes), the per-day minutes used, and the
    trajectory cache. Adding, dropping, or moving a session changes
    only the involved topics' contributions, so every candidate gain
    is computed from those topics alone and never needs a full
    objective recompute. ``evaluations`` counts candidate gains.
    """

    def __init__(self, topics, plan=None, as_of=None, model=None,
                 session_min=25):
        self.topics = sorted(topics, key=lambda t: t.name)
        self.by_name = {t.name: t for t in self.topics}
        self.as_of = as_of
        self.model = model
        self.session_min = session_min
        self.cache = _ObjectiveCache()
        self.plan: dict[str, list[int]] = {
            k: sorted(v) for k, v in (plan or {}).items()
        }
        for tp in self.topics:
            self.plan.setdefault(tp.name, [])
        self.contrib = {
            tp.name: self.cache.contribution(
                tp, self.plan[tp.name], as_of, model)
            for tp in self.topics
        }
        self.used: dict[int, int] = {}
        for days in self.plan.values():
            for d in days:
                self.used[d] = self.used.get(d, 0) + session_min
        self.evaluations = 0

    def total(self) -> float:
        return sum(self.contrib.values())

    def _refresh(self, name: str) -> None:
        self.contrib[name] = self.cache.contribution(
            self.by_name[name], self.plan[name], self.as_of, self.model)

    def add_gain(self, name: str, day: int) -> float:
        self.evaluations += 1
        tp = self.by_name[name]
        days = sorted(self.plan[name] + [day])
        return (self.cache.contribution(tp, days, self.as_of, self.model)
                - self.contrib[name])

    def drop_gain(self, name: str, day: int) -> float:
        self.evaluations += 1
        tp = self.by_name[name]
        days = list(self.plan[name])
        days.remove(day)
        return (self.cache.contribution(tp, days, self.as_of, self.model)
                - self.contrib[name])

    def move_gain(self, name: str, day: int, name2: str, day2: int) -> float:
        self.evaluations += 1
        if name2 == name:
            tp = self.by_name[name]
            days = list(self.plan[name])
            days.remove(day)
            days.append(day2)
            return (self.cache.contribution(tp, days, self.as_of,
                                           self.model)
                    - self.contrib[name])
        tp = self.by_name[name]
        tp2 = self.by_name[name2]
        days = list(self.plan[name])
        days.remove(day)
        days2 = sorted(self.plan[name2] + [day2])
        return ((self.cache.contribution(tp, days, self.as_of, self.model)
                 - self.contrib[name])
                + (self.cache.contribution(tp2, days2, self.as_of,
                                          self.model)
                   - self.contrib[name2]))

    def swap_gain(self, n1: str, d1: int, n2: str, d2: int) -> float:
        """Gain of exchanging the days of two sessions at once."""
        self.evaluations += 1
        tp1 = self.by_name[n1]
        tp2 = self.by_name[n2]
        days1 = list(self.plan[n1])
        days1.remove(d1)
        days1.append(d2)
        days2 = list(self.plan[n2])
        days2.remove(d2)
        days2.append(d1)
        return ((self.cache.contribution(tp1, days1, self.as_of,
                                        self.model)
                 - self.contrib[n1])
                + (self.cache.contribution(tp2, days2, self.as_of,
                                          self.model)
                   - self.contrib[n2]))

    def apply_add(self, name: str, day: int) -> None:
        self.plan[name].append(day)
        self.plan[name].sort()
        self._refresh(name)
        self.used[day] = self.used.get(day, 0) + self.session_min

    def apply_drop(self, name: str, day: int) -> None:
        self.plan[name].remove(day)
        self._refresh(name)
        self.used[day] -= self.session_min
        if self.used[day] <= 0:
            del self.used[day]

    def apply_move(self, name: str, day: int, name2: str, day2: int) -> None:
        self.apply_drop(name, day)
        self.apply_add(name2, day2)

    def apply_swap(self, n1: str, d1: int, n2: str, d2: int) -> None:
        self.plan[n1].remove(d1)
        self.plan[n2].remove(d2)
        self.plan[n1].append(d2)
        self.plan[n2].append(d1)
        self.plan[n1].sort()
        self.plan[n2].sort()
        self._refresh(n1)
        self._refresh(n2)
        # per-day totals are identical after a swap, so used is unchanged

    def reaugment(self, horizon, daily_cap, session_min=None,
                  rest_days=(), start_day=0, caps=None) -> int:
        """Greedily add the best positive-gain session until none helps.

        Deterministic: the same topic/day order and tie rule as
        greedy. Returns the number of sessions added.
        """
        if session_min is None:
            session_min = self.session_min
        self.session_min = session_min
        return _add_best_loop(self, horizon, daily_cap, session_min,
                              set(rest_days), start_day, caps)


def _add_best_loop(run, horizon, daily_cap, session_min, rest_days,
                   start_day, caps=None):
    """Add the best positive-gain session until none helps.

    The shared add loop for greedy and re-augmentation: topics in
    name order, days ascending, ties keep the first candidate whose
    gain clears the current best by more than MIN_GAIN. Returns the
    number of sessions added.
    """
    added = 0
    while True:
        best = None
        for tp in run.topics:
            last_day = min(horizon, tp.latest_exam_day)
            for d in range(start_day, last_day):
                cap = caps.get(d, daily_cap) if caps else daily_cap
                if not _day_has_room(run.used, d, session_min, cap,
                                     rest_days):
                    continue
                if run.plan[tp.name].count(d) >= MAX_PER_TOPIC_PER_DAY:
                    continue
                gain = run.add_gain(tp.name, d)
                if gain > MIN_GAIN and (best is None
                                        or gain > best[0] + MIN_GAIN):
                    best = (gain, tp.name, d)
        if best is None:
            return added
        _, name, d = best
        run.apply_add(name, d)
        added += 1


def _local_search_loop(run, horizon, daily_cap, session_min, rest_days,
                       start_day=0, max_passes=MAX_PASSES, caps=None):
    """Best-improvement moves until convergence, on ``run``.

    After each accepted relocation the freed capacity is refilled by
    deterministic re-augmentation, so a move never leaves usable
    minutes empty. Drops never open a useful slot on their own, but
    the same refill runs after them for uniformity.
    """

    def cap_of(day):
        return caps.get(day, daily_cap) if caps else daily_cap

    for _ in range(max_passes):
        used = dict(run.used)
        best = None
        sessions = sorted((n, d) for n, ds in run.plan.items()
                          for d in ds if n in run.by_name)
        for name, d in sessions:
            # candidate 1: drop the session
            gain = run.drop_gain(name, d)
            if gain > MIN_GAIN and (best is None
                                    or gain > best[0] + MIN_GAIN):
                best = (gain, ("drop", name, d))
            # candidate 2: move it to another (topic, day)
            for tp2 in run.topics:
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
                        cnt = run.plan[name].count(d2)
                    else:
                        cnt = run.plan[tp2.name].count(d2)
                    if cnt >= MAX_PER_TOPIC_PER_DAY:
                        continue
                    gain = run.move_gain(name, d, tp2.name, d2)
                    if gain > MIN_GAIN and (best is None
                                            or gain > best[0] + MIN_GAIN):
                        best = (gain, ("move", name, d, tp2.name, d2))
        if best is None:
            return
        _, action = best
        if action[0] == "drop":
            _, name, d = action
            run.apply_drop(name, d)
        else:
            _, name, d, name2, d2 = action
            run.apply_move(name, d, name2, d2)
        run.reaugment(horizon, daily_cap, session_min, rest_days,
                      start_day, caps=caps)


def _exchange_search(run, max_rounds=MAX_EXCHANGE_ROUNDS,
                     max_evals=MAX_EXCHANGE_EVALS):
    """Apply improving two-session swaps until none remains.

    A swap exchanges the days of two sessions from different topics.
    Per-day totals never change, so daily caps are preserved; the
    per-topic-per-day cap and each topic's exam day are checked
    explicitly. Only the two topics involved change contribution, so
    the gain is an exact affected-topic delta. Deterministic order:
    sessions sorted by (topic, day), pairs in lexicographic order,
    the single best swap per round, ties keep the first. Returns
    (rounds, swaps).
    """
    rounds = 0
    swaps = 0
    for _ in range(max_rounds):
        sessions = sorted((n, d) for n, ds in run.plan.items()
                          for d in ds if n in run.by_name)
        best = None
        evaluated = 0
        stop = False
        for i in range(len(sessions)):
            for j in range(i + 1, len(sessions)):
                if evaluated >= max_evals:
                    stop = True
                    break
                (n1, d1), (n2, d2) = sessions[i], sessions[j]
                if n1 == n2 or d1 == d2:
                    continue
                if run.plan[n1].count(d2) >= MAX_PER_TOPIC_PER_DAY:
                    continue
                if run.plan[n2].count(d1) >= MAX_PER_TOPIC_PER_DAY:
                    continue
                if d2 >= run.by_name[n1].latest_exam_day:
                    continue
                if d1 >= run.by_name[n2].latest_exam_day:
                    continue
                evaluated += 1
                gain = run.swap_gain(n1, d1, n2, d2)
                if gain > MIN_GAIN and (best is None
                                        or gain > best[0] + MIN_GAIN):
                    best = (gain, n1, d1, n2, d2)
            if stop:
                break
        if best is None:
            break
        _, n1, d1, n2, d2 = best
        run.apply_swap(n1, d1, n2, d2)
        rounds += 1
        swaps += 1
    return rounds, swaps


def oracle_gap(topics, plan, horizon, session_min, rest_days,
               start_day=0, as_of=None, model=None):
    """Reference oracle gap for a plan.

    The oracle is the sum over topics of the solo greedy objective:
    each topic planned alone with an unbounded daily cap, so it never
    competes for shared minutes. It is a reference, not a rigorous
    upper bound: solo greedy is itself heuristic, so the gap can in
    principle be negative on unusual inputs, and it is reported as
    measured. Returns (oracle, achieved, gap).
    """
    achieved = objective(topics, plan, as_of=as_of, model=model)
    oracle = 0.0
    for tp in topics:
        solo = greedy([tp], horizon, daily_cap=10 ** 9,
                      session_min=session_min, rest_days=set(rest_days),
                      start_day=start_day, as_of=as_of, model=model)
        oracle += objective([tp], solo, as_of=as_of, model=model)
    return oracle, achieved, oracle - achieved


def greedy(
    topics: list[PlanTopic],
    horizon: int,
    daily_cap: int,
    session_min: int,
    rest_days: set[int],
    start_day: int = 0,
    as_of: int | None = None,
    model=None,
    caps: dict[int, int] | None = None,
) -> dict[str, list[int]]:
    """Greedily add the highest marginal gain session until none helps.

    Delta-evaluated: adding a session to one topic changes only that
    topic's contribution, so each candidate's gain comes from the
    trajectory cache instead of a full objective recompute. Selection
    order and tie-breaking are unchanged, so the plan is identical to
    the full-recompute version. `caps` optionally overrides the daily
    cap per day (used by what-if replanning).
    """
    _validate_schedule_args(topics, horizon, daily_cap, session_min,
                            rest_days, start_day, as_of=as_of, caps=caps)
    run = _SearchRun(topics, as_of=as_of, model=model,
                     session_min=session_min)
    _add_best_loop(run, horizon, daily_cap, session_min, set(rest_days),
                   start_day, caps)
    return run.plan


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
    model=None,
) -> dict[str, list[int]]:
    """Best improvement moves until convergence. Deterministic.

    Delta-evaluated: a drop or move changes only the involved
    topics' contributions. After each accepted relocation the freed
    capacity is refilled by deterministic re-augmentation, so a move
    never leaves usable minutes empty.
    """
    _validate_schedule_args(topics, horizon, daily_cap, session_min,
                            rest_days, start_day, as_of=as_of, caps=caps,
                            max_passes=max_passes)
    for name, days in plan.items():
        for d in days:
            dv = _require_int(d, f"planned day for topic {name!r}")
            if dv < 0:
                raise ValueError(
                    f"planned day for topic {name!r} must be 0 or later, "
                    f"got {dv}"
                )
    run = _SearchRun(topics, plan=plan, as_of=as_of, model=model,
                     session_min=session_min)
    _local_search_loop(run, horizon, daily_cap, session_min,
                       set(rest_days), start_day, max_passes, caps)
    return run.plan


def build_plan(
    topics: list[PlanTopic],
    horizon: int,
    daily_cap: int,
    session_min: int,
    rest_days: set[int] | None = None,
    start_day: int = 0,
    seed: int = 0,
    as_of: int | None = None,
    model=None,
    stats: dict | None = None,
    caps: dict[int, int] | None = None,
) -> tuple[dict[str, list[int]], float]:
    """Greedy allocation followed by local search.

    Returns (plan, objective value). Deterministic for a given seed;
    the seed is accepted for interface stability and the algorithm uses
    no randomness. After local search, a bounded two-session exchange
    phase runs when the oracle gap justifies it, followed by a settle
    pass of local search plus re-augmentation. `stats`, when given,
    is filled with the run's cache counters, evaluation counts, the
    oracle gap, and the exchange rounds applied.

    `caps`, when given, maps day -> daily minutes for that day and
    overrides `daily_cap` on those days (the what-if path). Every
    phase of the search, greedy, local search, exchange, and settle,
    runs under the same caps, so a what-if plan and the standard plan
    are built by the same pipeline.

    The objective is evaluated as of `as_of` (default: `start_day`,
    the day planning starts): only exams still in the future at that
    day participate, so replanning never counts or chases a past
    exam's contribution. `model` selects the memory model; with none
    given each topic's own model applies, else the default.
    """
    _ = seed
    _validate_schedule_args(topics, horizon, daily_cap, session_min,
                            rest_days, start_day, as_of=as_of)
    rest = set(rest_days or ())
    if as_of is None:
        as_of = start_day
    run = _SearchRun(topics, as_of=as_of, model=model,
                     session_min=session_min)
    _add_best_loop(run, horizon, daily_cap, session_min, rest, start_day,
                   caps=caps)
    _local_search_loop(run, horizon, daily_cap, session_min, rest,
                       start_day, caps=caps)
    oracle, achieved, gap = oracle_gap(
        topics, run.plan, horizon, session_min, rest, start_day,
        as_of=as_of, model=model)
    exchanged = (0, 0)
    if oracle > 0 and gap / oracle > EXCHANGE_GAP_THRESHOLD:
        exchanged = _exchange_search(run)
        for _ in range(MAX_SETTLE_ROUNDS):
            before = {k: list(v) for k, v in run.plan.items()}
            _local_search_loop(run, horizon, daily_cap, session_min,
                               rest, start_day, caps=caps)
            added = run.reaugment(horizon, daily_cap, session_min, rest,
                                  start_day, caps=caps)
            if run.plan == before and added == 0:
                break
    if stats is not None:
        stats.update({
            "cache_hits": run.cache.hits,
            "cache_misses": run.cache.misses,
            "cache_entries": run.cache.entries,
            "evaluations": run.evaluations,
            "oracle": oracle,
            "oracle_gap": gap,
            "exchange_rounds": exchanged[0],
            "exchange_swaps": exchanged[1],
        })
    return run.plan, objective(topics, run.plan, as_of=as_of, model=model)


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
