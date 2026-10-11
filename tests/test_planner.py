import pytest

from studyforge.planner import (
    MAX_PER_TOPIC_PER_DAY,
    PlanTopic,
    build_plan,
    greedy,
    local_search,
    objective,
    simulate,
    to_schedule,
)


def make_topics():
    return [
        PlanTopic("Alpha", weight=0.5, exam_day=7, S=1.5, last_review=None),
        PlanTopic("Beta", weight=0.3, exam_day=14, S=1.0, last_review=None),
        PlanTopic("Gamma", weight=0.2, exam_day=21, S=0.75, last_review=None),
    ]


def test_simulate_no_sessions_is_baseline():
    tp = PlanTopic("A", 0.5, 7, 1.5, None)
    S, R = simulate(tp, [])
    assert S == pytest.approx(1.5)
    assert R == 0.0  # never studied: zero recall


def test_simulate_with_prior_review_uses_curve():
    import math

    tp = PlanTopic("A", 0.5, 7, 1.5, last_review=0)
    _, R = simulate(tp, [])
    assert R == pytest.approx(math.exp(-7 / 1.5))


def test_simulate_review_raises_recall_at_exam():
    tp = PlanTopic("A", 0.5, 7, 1.5, None)
    _, r0 = simulate(tp, [])
    _, r1 = simulate(tp, [3])
    assert r1 > r0


def test_plan_respects_daily_cap():
    topics = make_topics()
    plan, _ = build_plan(topics, horizon=21, daily_cap=50, session_min=25,
                         rest_days=set(), seed=0)
    sched = to_schedule(plan, 25)
    for day, sessions in sched.items():
        assert sum(m for _, m in sessions) <= 50


def test_plan_respects_rest_days():
    topics = make_topics()
    plan, _ = build_plan(topics, horizon=21, daily_cap=120, session_min=25,
                         rest_days={6, 13}, seed=0)
    sched = to_schedule(plan, 25)
    assert 6 not in sched
    assert 13 not in sched


def test_no_sessions_on_or_after_exam_day():
    topics = make_topics()
    plan, _ = build_plan(topics, horizon=21, daily_cap=120, session_min=25,
                         rest_days=set(), seed=0)
    by_name = {t.name: t for t in topics}
    for name, days in plan.items():
        for d in days:
            assert d < by_name[name].exam_day


def test_max_two_sessions_per_topic_per_day():
    topics = make_topics()
    plan, _ = build_plan(topics, horizon=21, daily_cap=200, session_min=25,
                         rest_days=set(), seed=0)
    for name, days in plan.items():
        for d in set(days):
            assert days.count(d) <= MAX_PER_TOPIC_PER_DAY


def test_all_topics_covered_with_generous_budget():
    topics = make_topics()
    plan, _ = build_plan(topics, horizon=21, daily_cap=120, session_min=25,
                         rest_days=set(), seed=0)
    for tp in topics:
        assert len(plan[tp.name]) >= 1


def test_deterministic_given_seed():
    topics = make_topics()
    kw = dict(horizon=21, daily_cap=120, session_min=25,
              rest_days={6}, seed=42)
    p1, o1 = build_plan(topics, **kw)
    p2, o2 = build_plan(topics, **kw)
    assert p1 == p2
    assert o1 == pytest.approx(o2)


def test_local_search_never_hurts():
    topics = make_topics()
    kw = dict(horizon=21, daily_cap=120, session_min=25, rest_days=set(),
              start_day=0)
    g = greedy(topics, **kw)
    improved = local_search(topics, g, **kw)
    assert objective(topics, improved) >= objective(topics, g) - 1e-9


def test_objective_increases_with_more_study_time():
    topics = make_topics()
    _, o_small = build_plan(topics, horizon=21, daily_cap=25, session_min=25,
                            rest_days=set(), seed=0)
    _, o_big = build_plan(topics, horizon=21, daily_cap=120, session_min=25,
                          rest_days=set(), seed=0)
    assert o_big >= o_small


def test_local_search_caps_respects_extra_day_cap():
    topics = make_topics()
    plan, _ = build_plan(topics, 14, 60, 30, set(), start_day=0)
    # no caps: uniform cap of 60 (two 30-min sessions max per day)
    plan2 = local_search(topics, plan, 14, 60, 30, set(), start_day=0)
    used = {}
    for days in plan2.values():
        for d in days:
            used[d] = used.get(d, 0) + 30
    assert all(v <= 60 for v in used.values())
    # with an extra 30 min on day 3, day 3 may hold three sessions
    plan3 = local_search(topics, plan, 14, 60, 30, set(), start_day=0,
                        caps={3: 90})
    used3 = {}
    for days in plan3.values():
        for d in days:
            used3[d] = used3.get(d, 0) + 30
    assert all(used3.get(d, 0) <= (90 if d == 3 else 60) for d in used3)
