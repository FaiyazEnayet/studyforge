"""Stage 3 A05: delta scoring, cached trajectories, deterministic search.

Written before the fix: the suite fails at collection on the missing
delta-scoring helpers, and the example-improvement test fails on the
pre-fix planner (objective 2.278051, no improving swap applied).
"""
import os
import subprocess
import sys

import pytest

from studyforge.planner import (
    MAX_PER_TOPIC_PER_DAY,
    MIN_GAIN,
    PlanTopic,
    _ObjectiveCache,
    _SearchRun,
    _exchange_search,
    build_plan,
    greedy,
    local_search,
    objective,
    oracle_gap,
    simulate,
)


def make_small():
    return [
        PlanTopic("Alpha", [("E", 0.5, 7)], S=1.5, last_review=None),
        PlanTopic("Beta", [("E", 0.3, 14)], S=1.0, last_review=None),
        PlanTopic("Gamma", [("E", 0.2, 21)], S=0.75, last_review=None),
    ]


def make_many():
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


def make_history():
    return [
        PlanTopic("Hist", [("E", 1.0, 10), ("F", 0.5, 20)], S=2.0,
                  last_review=4, difficulty=3,
                  history=[(0, None), (2, 0.8), (4, 0.6)]),
        PlanTopic("Fresh", [("E", 0.5, 10)], S=1.0, last_review=None),
    ]


def _ref_greedy(topics, horizon, daily_cap, session_min, rest_days,
                start_day=0, as_of=None, model=None, caps=None):
    """Independent full-recompute greedy: the pre-A05 algorithm."""
    from studyforge.planner import _day_has_room, _validate_schedule_args

    _validate_schedule_args(topics, horizon, daily_cap, session_min,
                            rest_days, start_day, as_of=as_of)
    plan = {tp.name: [] for tp in topics}
    used = {}
    ordered = sorted(topics, key=lambda t: t.name)

    def cap_of(day):
        return caps.get(day, daily_cap) if caps else daily_cap

    while True:
        base = objective(topics, plan, as_of=as_of, model=model)
        best = None
        for tp in ordered:
            last_day = min(horizon, tp.latest_exam_day)
            for d in range(start_day, last_day):
                if not _day_has_room(used, d, session_min, cap_of(d),
                                     rest_days):
                    continue
                if plan[tp.name].count(d) >= MAX_PER_TOPIC_PER_DAY:
                    continue
                plan[tp.name].append(d)
                gain = objective(topics, plan, as_of=as_of,
                                 model=model) - base
                plan[tp.name].pop()
                if gain > MIN_GAIN and (best is None
                                        or gain > best[0] + MIN_GAIN):
                    best = (gain, tp.name, d)
        if best is None:
            break
        _, name, d = best
        plan[name].append(d)
        plan[name].sort()
        used[d] = used.get(d, 0) + session_min
    return plan


def _ref_local_search(topics, plan, horizon, daily_cap, session_min,
                      rest_days, start_day=0, max_passes=100,
                      caps=None, as_of=None, model=None):
    """Independent full-recompute local search: the pre-A05 algorithm."""
    from studyforge.planner import _validate_schedule_args

    _validate_schedule_args(topics, horizon, daily_cap, session_min,
                            rest_days, start_day, as_of=as_of, caps=caps,
                            max_passes=max_passes)
    plan = {k: sorted(v) for k, v in plan.items()}
    ordered = sorted(topics, key=lambda t: t.name)

    def cap_of(day):
        return caps.get(day, daily_cap) if caps else daily_cap

    def used_minutes(p):
        u = {}
        for days in p.values():
            for d in days:
                u[d] = u.get(d, 0) + session_min
        return u

    for _ in range(max_passes):
        base = objective(topics, plan, as_of=as_of, model=model)
        used = used_minutes(plan)
        best = None
        sessions = sorted((n, d) for n, ds in plan.items() for d in ds)
        for name, d in sessions:
            plan[name].remove(d)
            gain = objective(topics, plan, as_of=as_of, model=model) - base
            plan[name].append(d)
            plan[name].sort()
            if gain > MIN_GAIN and (best is None
                                    or gain > best[0] + MIN_GAIN):
                best = (gain, ("drop", name, d))
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
                    gain = objective(topics, plan, as_of=as_of,
                                     model=model) - base
                    plan[tp2.name].remove(d2)
                    plan[name].append(d)
                    plan[name].sort()
                    if gain > MIN_GAIN and (best is None
                                            or gain > best[0] + MIN_GAIN):
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


def _check_valid(topics, plan, horizon, daily_cap, session_min,
                 rest_days, caps=None):
    by_name = {t.name: t for t in topics}
    used = {}
    for name, days in plan.items():
        assert name in by_name
        for d in days:
            assert d not in rest_days
            assert d < by_name[name].latest_exam_day
            assert days.count(d) <= MAX_PER_TOPIC_PER_DAY
            used[d] = used.get(d, 0) + session_min
    for d, minutes in used.items():
        cap = caps.get(d, daily_cap) if caps else daily_cap
        assert minutes <= cap


def test_stage3_studyforge_05_delta_equals_full_recompute():
    for topics in (make_small(), make_many(), make_history()):
        kw = dict(horizon=21, daily_cap=60, session_min=25,
                  rest_days={6}, start_day=0)
        plans = [
            {t.name: [] for t in topics},
            {t.name: [1] if t.name in ("Alpha", "T0", "Hist") else []
             for t in topics},
            _ref_greedy(topics, **kw),
        ]
        for plan in plans:
            run = _SearchRun(topics, plan=plan, session_min=25)
            assert run.total() == pytest.approx(
                objective(topics, plan), abs=1e-9)
            # every candidate add / drop / move gain matches full recompute
            base = objective(topics, plan)
            for tp in topics:
                for d in range(0, min(21, tp.latest_exam_day)):
                    if d in kw["rest_days"]:
                        continue
                    trial = {k: list(v) for k, v in plan.items()}
                    trial[tp.name].append(d)
                    trial[tp.name].sort()
                    want = objective(topics, trial) - base
                    assert run.add_gain(tp.name, d) == pytest.approx(
                        want, abs=1e-9)
            for name, days in plan.items():
                for d in set(days):
                    trial = {k: list(v) for k, v in plan.items()}
                    trial[name].remove(d)
                    want = objective(topics, trial) - base
                    assert run.drop_gain(name, d) == pytest.approx(
                        want, abs=1e-9)
            sessions = sorted((n, d) for n, ds in plan.items()
                              for d in ds)
            for (n1, d1), (n2, d2) in zip(sessions, sessions[1:]):
                if n1 == n2 or d1 == d2:
                    continue
                trial = {k: list(v) for k, v in plan.items()}
                trial[n1].remove(d1)
                trial[n2].remove(d2)
                trial[n1].append(d2)
                trial[n2].append(d1)
                want = objective(topics, trial) - base
                assert run.swap_gain(n1, d1, n2, d2) == pytest.approx(
                    want, abs=1e-9)


def test_stage3_studyforge_05_greedy_matches_full_recompute_oracle():
    for topics in (make_small(), make_many(), make_history()):
        kw = dict(horizon=21, daily_cap=60, session_min=25,
                  rest_days={6}, start_day=0)
        assert greedy(topics, **kw) == _ref_greedy(topics, **kw)


def test_stage3_studyforge_05_local_search_delta_matches_oracle_gains():
    # Every drop/move gain the delta search considers equals the
    # full-recompute gain, on the greedy plan of each fixture.
    for topics in (make_small(), make_many(), make_history()):
        kw = dict(horizon=21, daily_cap=60, session_min=25,
                  rest_days={6}, start_day=0)
        plan = _ref_greedy(topics, **kw)
        run = _SearchRun(topics, plan=plan, session_min=25)
        base = objective(topics, plan)
        sessions = sorted((n, d) for n, ds in plan.items() for d in ds)
        for name, d in sessions:
            trial = {k: list(v) for k, v in plan.items()}
            trial[name].remove(d)
            assert run.drop_gain(name, d) == pytest.approx(
                objective(topics, trial) - base, abs=1e-9)
        by_name = {t.name: t for t in topics}
        for name, d in sessions:
            for tp2 in sorted(topics, key=lambda t: t.name):
                for d2 in range(0, min(21, tp2.latest_exam_day)):
                    if (tp2.name, d2) == (name, d) or d2 in kw["rest_days"]:
                        continue
                    trial = {k: list(v) for k, v in plan.items()}
                    trial[name].remove(d)
                    trial[tp2.name].append(d2)
                    trial[tp2.name].sort()
                    assert run.move_gain(name, d, tp2.name, d2) == \
                        pytest.approx(
                            objective(topics, trial) - base, abs=1e-9)
        _ = by_name


def test_stage3_studyforge_05_cache_invalidation_recomputes_identically():
    topics = make_history()
    cache = _ObjectiveCache()
    tp = topics[0]
    first = cache.contribution(tp, [1, 3, 5])
    assert cache.hits == 0
    second = cache.contribution(tp, [1, 3, 5])
    assert second == first
    assert cache.hits >= 1
    cache.invalidate(tp.name)
    third = cache.contribution(tp, [1, 3, 5])
    assert third == first
    cache.invalidate()
    fourth = cache.contribution(tp, [1, 3, 5])
    assert fourth == first
    # the cached simulate matches the direct one exactly
    S, R = cache.simulate(tp, [1, 3, 5])
    S2, R2 = simulate(tp, [1, 3, 5])
    assert S == S2
    assert R == R2


def test_stage3_studyforge_05_reaugmentation_refills_opened_capacity():
    topics = make_small()
    kw = dict(horizon=21, daily_cap=60, session_min=25,
              rest_days={6}, start_day=0)
    plan = greedy(topics, **kw)
    assert plan["Alpha"]
    # free capacity by removing one session: re-augmentation must
    # find a strictly positive gain and add a session back
    slim = {k: list(v) for k, v in plan.items()}
    slim["Alpha"].pop(0)
    run = _SearchRun(topics, plan=slim, session_min=25)
    before = run.total()
    added = run.reaugment(horizon=21, daily_cap=60, session_min=25,
                          rest_days={6}, start_day=0)
    assert added >= 1
    assert run.total() > before + MIN_GAIN
    _check_valid(topics, run.plan, 21, 60, 25, {6})
    # a converged build_plan is augmentation-closed: nothing to add
    done, _ = build_plan(topics, **kw)
    run2 = _SearchRun(topics, plan=done, session_min=25)
    assert run2.reaugment(horizon=21, daily_cap=60, session_min=25,
                           rest_days={6}, start_day=0) == 0


def test_stage3_studyforge_05_exchange_search_keeps_valid_and_improves():
    for topics in (make_small(), make_many()):
        kw = dict(horizon=21, daily_cap=60, session_min=25,
                  rest_days={6}, start_day=0)
        plan = _ref_local_search(topics, _ref_greedy(topics, **kw), **kw)
        run = _SearchRun(topics, plan=plan, session_min=25)
        before = run.total()
        rounds, swaps = _exchange_search(run)
        assert run.total() >= before - 1e-9
        _check_valid(topics, run.plan, 21, 60, 25, {6})
        # after the exchange search, no single swap improves (checked
        # against full recomputation on these small fixtures)
        base = objective(topics, run.plan)
        sessions = sorted((n, d) for n, ds in run.plan.items()
                          for d in ds)
        for i in range(len(sessions)):
            for j in range(i + 1, len(sessions)):
                (n1, d1), (n2, d2) = sessions[i], sessions[j]
                if n1 == n2 or d1 == d2:
                    continue
                trial = {k: list(v) for k, v in run.plan.items()}
                trial[n1].remove(d1)
                trial[n2].remove(d2)
                if trial[n1].count(d2) >= MAX_PER_TOPIC_PER_DAY:
                    continue
                if trial[n2].count(d1) >= MAX_PER_TOPIC_PER_DAY:
                    continue
                trial[n1].append(d2)
                trial[n2].append(d1)
                assert objective(topics, trial) - base <= MIN_GAIN
        assert rounds >= 0 and swaps >= 0


def test_stage3_studyforge_05_build_plan_improves_example():
    from studyforge import state as st
    from studyforge.cli import _plan_topics

    root = os.path.join(os.path.dirname(__file__), "..")
    cfg = st.load_config(os.path.join(root, "exams.yaml"))
    state = st.new_state(cfg)
    topics = _plan_topics(state)
    horizon = max(t.latest_exam_day for t in topics)
    plan, obj = build_plan(
        topics, horizon, cfg["daily_budget_minutes"],
        cfg["session_minutes"], set(cfg["rest_days"]),
        start_day=0, seed=0,
    )
    # the pre-A05 plan is 72 sessions at objective 2.278051; the
    # exchange phase must find the improving swaps the old search
    # could not see (single moves only), so the objective rises to
    # the measured 2.312646 with the same 72 sessions
    assert obj > 2.278051
    assert sum(len(v) for v in plan.values()) == 72
    _check_valid(topics, plan, horizon, cfg["daily_budget_minutes"],
                 cfg["session_minutes"], set(cfg["rest_days"]))
    # and the new value is pinned exactly
    assert obj == pytest.approx(2.312646, abs=1e-6)


def test_stage3_studyforge_05_byte_identical_plans_in_fresh_processes():
    root = os.path.join(os.path.dirname(__file__), "..")
    driver = (
        "import json, os, sys\n"
        "sys.path.insert(0, os.path.join(os.getcwd(), 'src'))\n"
        "from studyforge import state as st\n"
        "from studyforge.cli import _plan_topics\n"
        "from studyforge.planner import build_plan\n"
        "cfg = st.load_config('exams.yaml')\n"
        "state = st.new_state(cfg)\n"
        "topics = _plan_topics(state)\n"
        "horizon = max(t.latest_exam_day for t in topics)\n"
        "plan, obj = build_plan(topics, horizon,"
        " cfg['daily_budget_minutes'], cfg['session_minutes'],"
        " set(cfg['rest_days']), start_day=0, seed=0)\n"
        "print(json.dumps({'plan': plan, 'objective': repr(obj)},"
        " sort_keys=True))\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(root, "src")
    env["TMPDIR"] = os.path.expanduser("~/tmp")
    outs = []
    for _ in range(2):
        proc = subprocess.run(
            [sys.executable, "-c", driver], capture_output=True, text=True,
            cwd=root, env=env, timeout=600)
        assert proc.returncode == 0, proc.stderr
        outs.append(proc.stdout.encode())
    assert outs[0] == outs[1]
    assert outs[0]


def test_stage3_studyforge_05_oracle_gap_values():
    for topics in (make_small(), make_many()):
        kw = dict(horizon=21, daily_cap=60, session_min=25,
                  rest_days={6}, start_day=0)
        plan = _ref_greedy(topics, **kw)
        oracle, achieved, gap = oracle_gap(
            topics, plan, horizon=21, session_min=25, rest_days={6},
            start_day=0)
        assert oracle >= achieved - 1e-9
        assert gap >= -1e-9
        assert gap == pytest.approx(oracle - achieved, abs=1e-9)
        # the oracle is the sum of per-topic solo greedy runs, on the
        # same rest days as the joint plan
        solo_total = 0.0
        for tp in topics:
            solo = _ref_greedy([tp], 21, 10 ** 9, 25, {6}, start_day=0)
            solo_total += objective([tp], solo)
        assert oracle == pytest.approx(solo_total, abs=1e-9)


def test_stage3_studyforge_05_greedy_caps_matches_what_if_reference():
    topics = make_many()
    caps = {3: 90, 5: 30, 9: 120}
    kw = dict(horizon=21, daily_cap=60, session_min=25,
              rest_days={6}, start_day=0, caps=caps)
    assert greedy(topics, **kw) == _ref_greedy(topics, **kw)


def test_stage3_studyforge_05_what_if_runs_full_pipeline():
    # the what-if path runs the same build_plan pipeline as the
    # standard plan, with per-day caps, so extra capacity can only
    # help: +30 minutes on day 3 of the example buys two more
    # sessions and a higher objective than the standard 2.312646
    from studyforge import cli, state as st

    cfg = st.load_config("exams.yaml")
    state = st.new_state(cfg)
    plan, obj, horizon = cli._rebuild(state, 0, 0, extra={3: 30})
    assert horizon == 21
    assert sum(len(v) for v in plan.values()) == 74
    assert obj == pytest.approx(2.330030, abs=1e-6)
    assert obj > 2.312646


def test_stage3_studyforge_05_benchmark_script_runs():
    root = os.path.join(os.path.dirname(__file__), "..")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(root, "src")
    env["TMPDIR"] = os.path.expanduser("~/tmp")
    proc = subprocess.run(
        [sys.executable, "benchmarks/planner_scaling.py", "--quick"],
        capture_output=True, text=True, cwd=root, env=env, timeout=600)
    assert proc.returncode == 0, proc.stderr
    assert "oracle gap" in proc.stdout.lower()
    assert "small" in proc.stdout.lower()
