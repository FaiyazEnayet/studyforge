"""Planner scaling benchmark (A05).

Memory, runtime, and oracle-gap numbers for build_plan on fixed,
documented test sets. The sets are literal below, so every run
measures the same work: no randomness, no network, no wall-clock
dependence in the plans themselves (only in the timings).

Sets
----
small    3 topics, horizon 21, daily cap 120, 25-minute sessions.
medium   8 topics, horizon 21, daily cap 60, 25-minute sessions.
large    12 topics, horizon 42, daily cap 120, 25-minute sessions.
example  the included exams.yaml, as the CLI builds it.

Columns
-------
sessions       sessions in the built plan.
objective      the plan's objective (full recompute).
oracle         the independent-topic oracle reference (see
               planner.oracle_gap): each topic planned alone with an
               unbounded daily cap.
oracle gap     oracle minus objective, and as a share of the oracle.
build time     wall seconds for build_plan.
simulates      trajectory-cache misses: the simulate computations the
               run actually performed.
counterfactual the simulate computations full recomputation would
               have needed for the same candidate evaluations
               (evaluations x topics), and the ratio between the two.
cache entries  trajectory-cache entries alive at the end of the run.
peak RSS       the process high-water RSS after the set, in MB.

Run: python benchmarks/planner_scaling.py [--quick]
``--quick`` runs only the small set, for smoke tests.
"""
from __future__ import annotations

import argparse
import os
import resource
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from studyforge import state as st  # noqa: E402
from studyforge.cli import _plan_topics  # noqa: E402
from studyforge.planner import PlanTopic, build_plan  # noqa: E402


def _small():
    return [
        PlanTopic("Alpha", [("E", 0.5, 7)], S=1.5, last_review=None),
        PlanTopic("Beta", [("E", 0.3, 14)], S=1.0, last_review=None),
        PlanTopic("Gamma", [("E", 0.2, 21)], S=0.75, last_review=None),
    ]


def _medium():
    topics = []
    for i, (exam, w, S) in enumerate([
        ("E1", 0.9, 2.0), ("E1", 0.7, 1.2), ("E2", 0.8, 1.5),
        ("E2", 0.4, 0.8), ("E3", 0.6, 2.5), ("E3", 0.5, 1.0),
        ("E1", 0.3, 3.0), ("E3", 0.2, 0.6),
    ]):
        day = {"E1": 7, "E2": 14, "E3": 21}[exam]
        topics.append(PlanTopic(f"T{i}", [(exam, w, day)], S=S,
                                last_review=None))
    return topics


def _large():
    topics = []
    for i in range(12):
        exam, day = [("E1", 14), ("E2", 28), ("E3", 42)][i % 3]
        topics.append(PlanTopic(
            f"L{i}", [(exam, 0.3 + 0.05 * (i % 5), day)],
            S=0.8 + 0.2 * (i % 6), last_review=None))
    return topics


def _example():
    cfg = st.load_config(os.path.join(ROOT, "exams.yaml"))
    state = st.new_state(cfg)
    topics = _plan_topics(state)
    horizon = max(t.latest_exam_day for t in topics)
    return topics, horizon, cfg["daily_budget_minutes"], \
        cfg["session_minutes"], set(cfg["rest_days"])


SETS = [
    ("small", lambda: (_small(), 21, 120, 25, set())),
    ("medium", lambda: (_medium(), 21, 60, 25, {6})),
    ("large", lambda: (_large(), 42, 120, 25, {7, 14, 21})),
    ("example", _example),
]


def _peak_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def run_set(name, topics, horizon, daily_cap, session_min, rest_days):
    stats: dict = {}
    start = time.perf_counter()
    plan, obj = build_plan(
        topics, horizon, daily_cap, session_min, rest_days,
        start_day=0, seed=0, stats=stats)
    elapsed = time.perf_counter() - start
    sessions = sum(len(v) for v in plan.values())
    oracle = stats["oracle"]
    gap = stats["oracle_gap"]
    rel_gap = gap / oracle if oracle else 0.0
    counterfactual = stats["evaluations"] * len(topics)
    ratio = (counterfactual / stats["cache_misses"]
             if stats["cache_misses"] else float("inf"))
    return {
        "set": name,
        "sessions": sessions,
        "objective": obj,
        "oracle": oracle,
        "gap": gap,
        "rel_gap": rel_gap,
        "time_s": elapsed,
        "simulates": stats["cache_misses"],
        "counterfactual": counterfactual,
        "ratio": ratio,
        "cache_entries": stats["cache_entries"],
        "cache_hits": stats["cache_hits"],
        "exchanges": stats["exchange_swaps"],
        "peak_rss_mb": _peak_rss_mb(),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="run only the small set")
    args = parser.parse_args(argv)
    chosen = SETS[:1] if args.quick else SETS
    rows = []
    for name, make in chosen:
        topics, horizon, cap, session_min, rest = make()
        rows.append(run_set(name, topics, horizon, cap, session_min, rest))
    print("planner scaling benchmark (A05): fixed test sets, "
          "deterministic plans")
    print(f"{'set':<8} {'sessions':>8} {'objective':>10} {'oracle':>8} "
          f"{'oracle gap':>18} {'build time':>11} {'simulates':>10} "
          f"{'counterfactual':>14} {'cache':>6} {'peak RSS':>9}")
    for r in rows:
        print(f"{r['set']:<8} {r['sessions']:>8} {r['objective']:>10.6f} "
              f"{r['oracle']:>8.4f} "
              f"{r['gap']:>10.4f} ({r['rel_gap']:>5.1%}) "
              f"{r['time_s']:>10.2f}s {r['simulates']:>10} "
              f"{r['counterfactual']:>14} {r['cache_entries']:>6} "
              f"{r['peak_rss_mb']:>8.1f} MB")
    print()
    for r in rows:
        print(f"{r['set']}: {r['sessions']} sessions, objective "
              f"{r['objective']:.6f}, oracle gap {r['gap']:.4f} "
              f"({r['rel_gap']:.1%} of the oracle), "
              f"{r['simulates']} trajectory computations against "
              f"{r['counterfactual']} for full recomputation "
              f"({r['ratio']:.1f}x fewer), "
              f"{r['cache_hits']} cache hits, "
              f"{r['exchanges']} exchange swaps, "
              f"{r['time_s']:.2f}s, peak RSS {r['peak_rss_mb']:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
