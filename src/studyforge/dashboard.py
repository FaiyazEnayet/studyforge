"""View-layer helpers for the dashboard.

Everything here is a thin layer over the real engine (memory, planner,
state, cli): no reimplemented math, no invented numbers. The dashboard
generator script and the tests both import from this module.

Topic schema (one entry per normalized topic ID in plan_data topic_info):
each entry carries its bindings as (exam name, weight, exam day)
tuples, exam_days listing every binding deadline, and exam_day as the
latest binding deadline, which is the single deadline the curve views
use. A topic bound to several exams keeps one memory state and one
entry; consumers must not invent a deadline that is not in bindings.
topic_exam_day validates that contract and fails loudly on a mismatch
instead of plotting against a wrong or missing deadline.
"""

from __future__ import annotations

import math

from . import cli as _cli
from .memory import TopicState, fit_stability, initial_stability
from .planner import PlanTopic, build_plan, simulate, to_schedule


def fresh_topics(config: dict) -> list[PlanTopic]:
    """PlanTopics for a brand-new plan (day zero, no reviews logged yet).

    Topics shared by several exams get one PlanTopic with all bindings.
    """
    from .state import topic_id
    by_id: dict[str, dict] = {}
    for ex in config["exams"]:
        for t in ex["topics"]:
            tid = topic_id(t["name"])
            e = by_id.setdefault(tid, {"difficulty": t["difficulty"], "bindings": []})
            e["bindings"].append((ex["name"], t["weight"], ex["day"]))
    return [
        PlanTopic(tid, e["bindings"], initial_stability(e["difficulty"]), None)
        for tid, e in by_id.items()
    ]


def topic_exam_day(info: dict) -> int:
    """Validated latest binding deadline for one topic_info entry.

    Returns info["exam_day"] only when it matches the latest day in
    info["bindings"] (and info["exam_days"] when present). Raises
    ValueError on a missing or inconsistent exam_day so a generator
    fails loudly instead of plotting against a wrong deadline.
    """
    if "exam_day" not in info:
        raise ValueError("topic_info entry lacks exam_day")
    exam_day = info["exam_day"]
    bindings = info.get("bindings", [])
    if bindings:
        latest = max(d for _, _, d in bindings)
        if exam_day != latest:
            raise ValueError(
                f"topic_info exam_day {exam_day!r} does not match "
                f"latest binding day {latest!r}")
    if "exam_days" in info:
        expected = sorted({d for _, _, d in bindings})
        if list(info["exam_days"]) != expected:
            raise ValueError(
                f"topic_info exam_days {info['exam_days']!r} does not "
                f"match binding days {expected!r}")
    return exam_day


def plan_data(config: dict, today: int = 0, seed: int = 0) -> dict:
    """Run the real planner and return everything the dashboard needs.

    Returns a dict with: plan (topic -> sorted days), objective,
    schedule (day -> [(topic, minutes)]), horizon, per-topic info
    (sessions, final stability, recall at exam, bindings, exam_days,
    and exam_day as the latest binding deadline), per-exam summary
    (sessions, minutes, weighted recall at exam after the plan),
    per-day minutes used, and session counts.
    """
    topics = fresh_topics(config)
    horizon = max(t.latest_exam_day for t in topics)
    rest = set(config["rest_days"])
    daily = config["daily_budget_minutes"]
    sess = config["session_minutes"]
    plan, obj = build_plan(topics, horizon, daily, sess, rest,
                           start_day=today, seed=seed)
    sched = to_schedule(plan, sess)
    by_name = {t.name: t for t in topics}
    from .state import topic_id as _tid
    exam_of: dict[str, list[str]] = {}
    for ex in config["exams"]:
        for t in ex["topics"]:
            exam_of.setdefault(_tid(t["name"]), []).append(ex["name"])

    topic_info = {}
    for name, days in plan.items():
        tp = by_name[name]
        S_final, recalls = simulate(tp, days)
        topic_info[name] = {
            "exams": exam_of[name],
            "sessions": len(days),
            "minutes": len(days) * sess,
            "days": sorted(days),
            "S_final": S_final,
            # recall per binding, in binding order
            "recalls": recalls,
            "bindings": tp.bindings,
            # every binding deadline, plus the latest one as exam_day
            "exam_days": sorted({d for _, _, d in tp.bindings}),
            "exam_day": tp.latest_exam_day,
            "S0": tp.S,
        }

    exam_summary: dict[str, dict] = {}
    for ex in config["exams"]:
        num = den = 0.0
        minutes = sessions = 0
        n_topics = 0
        for name, info in topic_info.items():
            for (b_ex, w, _), R in zip(info["bindings"], info["recalls"]):
                if b_ex != ex["name"] or R is None:
                    continue
                num += w * R
                den += w
                n_topics += 1
            if ex["name"] in info["exams"]:
                # Credit only sessions held before this exam's
                # deadline: a shared topic's later sessions serve its
                # later exams, not this one.
                before = sum(1 for d in info["days"] if d < ex["day"])
                sessions += before
                minutes += before * sess
        exam_summary[ex["name"]] = {
            "day": ex["day"],
            "topics": n_topics,
            "sessions": sessions,
            "minutes": minutes,
            "recall_at_exam": num / den if den else 0.0,
        }

    day_minutes = {d: sum(m for _, m in sessions)
                   for d, sessions in sched.items()}
    n_sessions = sum(len(v) for v in plan.values())
    return {
        "config": config,
        "plan": plan,
        "objective": obj,
        "schedule": sched,
        "horizon": horizon,
        "topic_info": topic_info,
        "exam_summary": exam_summary,
        "day_minutes": day_minutes,
        "n_sessions": n_sessions,
        "total_minutes": n_sessions * sess,
        "exam_of": exam_of,
    }


def recall_trajectory(name: str, plan_days: list[int], S0: float,
                      exam_day: int, horizon: int,
                      last_review: int | None = None,
                      resolution: int = 4) -> dict:
    """Day-by-day recall curve for one topic under its planned reviews.

    Walks the real TopicState forward: records recall each day, then
    applies the review update on review days. Returns xs, recall ys,
    the review day markers, and the final stability.
    """
    ts = TopicState(name=name, S=S0, last_review=last_review)
    review_days = set(plan_days)
    xs: list[float] = []
    ys: list[float] = []
    marks: list[tuple[float, float]] = []
    for i in range(horizon * resolution + 1):
        day = i / resolution
        xs.append(day)
        ys.append(ts.recall_on(day))
        # Reviews land on whole days: record the pre-review recall,
        # then apply the update. The mark sits at the dip where the
        # review actually happened.
        if day == int(day) and int(day) in review_days:
            marks.append((day, ts.recall_on(day)))
            ts.review(int(day))
    return {
        "xs": xs,
        "ys": ys,
        "review_marks": marks,
        "S_final": ts.S,
        "recall_at_exam": ts.recall_on(exam_day),
    }


def _state_from_config(config: dict) -> dict:
    """Minimal CLI state dict for a fresh plan, for the what-if rebuild."""
    from .state import new_state
    return new_state(config)


def what_if_data(config: dict, today: int, day: int,
                 extra_minutes: int, seed: int = 0) -> dict:
    """Baseline plan vs plan with extra minutes on one day.

    Uses the same rebuild path as the CLI. Returns objectives, the
    objective delta, and the added/displaced session lists.
    """
    state = _state_from_config(config)
    topics = _cli._plan_topics(state)
    horizon = max(t.latest_exam_day for t in topics)
    rest = set(config["rest_days"])
    base_plan, base_obj = build_plan(
        topics, horizon, config["daily_budget_minutes"],
        config["session_minutes"], rest, start_day=today, seed=seed,
    )
    new_plan, new_obj, _ = _cli._rebuild(
        state, today, seed, extra={day: extra_minutes})
    base_sched = to_schedule(base_plan, config["session_minutes"])
    new_sched = to_schedule(new_plan, config["session_minutes"])
    added, removed = _cli._diff_plans(base_sched, new_sched, today)
    from .state import topic_id
    exam_of = {}
    for ex in config["exams"]:
        for t in ex["topics"]:
            exam_of.setdefault(topic_id(t["name"]), []).append(ex["name"])
    exam_str = {tid: ",".join(names) for tid, names in exam_of.items()}
    return {
        "day": day,
        "extra_minutes": extra_minutes,
        "baseline_obj": base_obj,
        "new_obj": new_obj,
        "delta": new_obj - base_obj,
        "added": [(d, exam_str[n], n, m) for d, n, m in added],
        "removed": [(d, exam_str[n], n, m) for d, n, m in removed],
    }


def fit_demo_data() -> dict:
    """The fitting demo: three illustrative (interval, recall) pairs.

    These are example points, not logged user history: no review
    records back them, and the provenance field says so explicitly.
    Returns the fitted stability, curve data for the chart, and that
    provenance metadata.
    """
    pairs = [(3.0, 0.45), (2.0, 0.30), (2.0, 0.14)]
    S_fit = fit_stability([(t, r) for t, r in pairs])
    assert S_fit is not None
    xs = [i * 0.25 for i in range(int(14 / 0.25) + 1)]
    curve = [math.exp(-x / S_fit) for x in xs]
    return {
        "pairs": pairs,
        "S_fit": S_fit,
        "xs": xs,
        "curve": curve,
        "provenance": {
            "kind": "illustrative_fixture",
            "source": "worked example pairs, not logged history",
            "logged_records": 0,
        },
    }
