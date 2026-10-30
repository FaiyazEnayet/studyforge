"""A01 regression fixture: as-of-exam evaluation and snapshot discipline.

Written against the defects named in handoffs section 146 BEFORE the
fix, so each test fails on the pre-fix code for the stated reason:

* planner.simulate evaluates an earlier exam from today's (S,
  last_review): a topic shared by a day-7 and a day-21 exam and
  reviewed on day 10 gets recall exp(+(10-7)/S) > 1 at the day-7 exam.
* state.readiness reports a past exam from today's state as well
  (the memory clamp turns that into a fabricated 100%-style figure).
* Forward replanning counts the past exam's contribution in the
  objective it optimizes.
* dashboard.plan_data credits every session of a shared topic to
  every bound exam, including sessions on/after the earlier exam's
  deadline that cannot serve it.

Expected values are hand-derived from the memory model in memory.py
(S0 = 3/difficulty; a review multiplies S by 1 + (1 - R); R = exp(-t/S)
at the review, 0 for the first review), not copied from the code under
test. Shared scenario: Probability (difficulty 3, so S0 = 1.0) bound to
a day-7 Midterm (weight 0.6) and a day-21 Final (weight 0.5), logged
reviews on days 3 and 10 with no observed recall:

    after day 3:  S = 1.0 * 2 = 2.0, last_review = 3
    day 10:       R = exp(-7/2) = 0.030197, S = 2 * (1 + 1 - R)
                  = 3.939605, last_review = 10
    as of day 7:  only the day-3 review exists, so recall at day 7
                  is exp(-(7-3)/2.0) = exp(-2) = 0.135335
    at day 21:    exp(-(21-10)/3.939605) = 0.061298
"""

import math

import pytest

from studyforge import cli as _cli
from studyforge import dashboard as dash
from studyforge import state as st
from studyforge.memory import TopicState
from studyforge.planner import objective, simulate

# Hand-derived scenario constants (see module docstring).
S_AFTER_DAY3 = 2.0
R_AT_DAY10 = math.exp(-7.0 / S_AFTER_DAY3)
S_AFTER_DAY10 = S_AFTER_DAY3 * (1.0 + (1.0 - R_AT_DAY10))
RECALL_MIDTERM_ASOF = math.exp(-(7 - 3) / S_AFTER_DAY3)  # exp(-2)
RECALL_FINAL = math.exp(-(21 - 10) / S_AFTER_DAY10)


def _shared_config():
    return {
        "exams": [
            {"name": "Midterm", "day": 7, "topics": [
                {"name": "Probability", "difficulty": 3, "weight": 0.6},
                {"name": "Algebra", "difficulty": 2, "weight": 0.4}]},
            {"name": "Final", "day": 21, "topics": [
                {"name": "Probability", "difficulty": 3, "weight": 0.5},
                {"name": "Calculus", "difficulty": 4, "weight": 0.5}]},
        ],
        "rest_days": [],
        "daily_budget_minutes": 120,
        "session_minutes": 25,
    }


def _log(state, tid, day):
    """Apply one review exactly as cli.cmd_log does (no observed recall)."""
    t = state["topics"][tid]
    ts = TopicState(name=tid, difficulty=t["difficulty"], S=t["S"],
                    last_review=t["last_review"])
    prev = ts.last_review
    ts.review(day)
    t["S"] = ts.S
    t["last_review"] = ts.last_review
    state["history"].append(
        {"day": day, "topic": tid, "minutes": 25,
         "interval": None if prev is None else day - prev, "recall": None})


def _state_a():
    """Shared config with Probability logged on days 3 and 10."""
    state = st.new_state(_shared_config())
    _log(state, "probability", 3)
    _log(state, "probability", 10)
    return state


def _prob(state):
    topics = _cli._plan_topics(state)
    return next(t for t in topics if t.name == "probability")


def test_stage3_studyforge_01_day7_day21_counterexample_asof_recall():
    prob = _prob(_state_a())
    _, (r_mid, r_fin) = simulate(prob, [])
    # The day-7 recall must come from the state as of day 7 (only the
    # day-3 review), not from today's state after the day-10 review.
    assert r_mid == pytest.approx(RECALL_MIDTERM_ASOF)
    assert r_fin == pytest.approx(RECALL_FINAL)
    assert 0.0 <= r_mid <= 1.0


def test_stage3_studyforge_01_no_reviews_before_exam_is_zero_not_clipped():
    state = st.new_state(_shared_config())
    _log(state, "probability", 10)
    prob = _prob(state)
    _, (r_mid, r_fin) = simulate(prob, [])
    # As of day 7 the topic was never reviewed: recall is exactly 0.
    # A clamp of the impossible exp(+3/2) would read 1.0 instead; the
    # fix must reconstruct, not clip.
    assert r_mid == 0.0
    assert r_mid != 1.0
    assert r_fin == pytest.approx(math.exp(-(21 - 10) / S_AFTER_DAY3))


def test_stage3_studyforge_01_past_exam_excluded_from_replan_objective():
    state = _state_a()
    plan, obj, _ = _cli._rebuild(state, today=10, seed=0)
    topics = _cli._plan_topics(state)
    # Nothing is scheduled before today, and a topic bound only to the
    # past exam gets no sessions at all.
    assert all(d >= 10 for days in plan.values() for d in days)
    assert plan["algebra"] == []
    assert sum(len(v) for v in plan.values()) > 0
    # The optimized objective counts only bindings whose exam is still
    # in the future at the as-of day.
    expected = 0.0
    for tp in topics:
        _, recalls = simulate(tp, plan[tp.name])
        for (_, w, exam_day), r in zip(tp.bindings, recalls):
            if exam_day > 10:
                expected += w * r
    assert obj == pytest.approx(expected)
    # The past binding still exists and still evaluates (as-of), so
    # the unrestricted objective is strictly larger: the exclusion is
    # a planning-horizon rule, not a deleted binding.
    assert objective(topics, plan) > obj


def test_stage3_studyforge_01_readiness_historical_ignores_future_events():
    state = _state_a()
    R = st.readiness(state, today=10)
    assert R["Midterm"]["readiness"] == pytest.approx(
        0.6 * RECALL_MIDTERM_ASOF)
    assert R["Final"]["readiness"] == pytest.approx(0.5 * RECALL_FINAL)
    # A later review (day 14, while preparing for the Final) must not
    # change the historical day-7 estimate.
    _log(state, "probability", 14)
    R2 = st.readiness(state, today=14)
    assert R2["Midterm"]["readiness"] == pytest.approx(
        0.6 * RECALL_MIDTERM_ASOF)


def test_stage3_studyforge_01_legacy_history_marks_past_exam_unavailable():
    state = _state_a()
    # A migrated legacy state: history recorded topics by display name,
    # so no ID-keyed events back the current (S, last_review). The
    # as-of day-7 state cannot be reconstructed and must be marked
    # unavailable rather than evaluated from today's state.
    for h in state["history"]:
        h["topic"] = "Probability"
    R = st.readiness(state, today=10)
    assert R["Midterm"]["readiness"] is None
    # The future exam still evaluates from the current state, and no
    # binding is lost in the reporting.
    assert R["Final"]["readiness"] == pytest.approx(0.5 * RECALL_FINAL)
    assert set(R) == {"Midterm", "Final"}
    assert R["Midterm"]["topics"] == 2


def test_stage3_studyforge_01_plan_data_attributes_only_before_deadline():
    cfg = {
        "daily_budget_minutes": 120, "session_minutes": 25, "rest_days": [],
        "exams": [
            {"name": "Midterm", "day": 7, "topics": [
                {"name": "Probability", "difficulty": 5, "weight": 0.5},
                {"name": "Algebra", "difficulty": 3, "weight": 0.5}]},
            {"name": "Final", "day": 21, "topics": [
                {"name": "Probability", "difficulty": 5, "weight": 0.5},
                {"name": "Calculus", "difficulty": 3, "weight": 0.5}]},
        ],
    }
    data = dash.plan_data(cfg)
    info = data["topic_info"]
    # Scenario validity: the shared topic really is studied after the
    # Midterm deadline (for the Final), so attribution can over-count.
    assert any(d >= 7 for d in info["probability"]["days"])
    for exam_name, day in (("Midterm", 7), ("Final", 21)):
        expected = sum(
            sum(1 for d in inf["days"] if d < day)
            for inf in info.values() if exam_name in inf["exams"])
        summary = data["exam_summary"][exam_name]
        assert summary["sessions"] == expected
        assert summary["minutes"] == expected * cfg["session_minutes"]


def test_stage3_studyforge_01_all_probabilities_finite_and_bounded():
    for state in (_state_a(),):
        prob = _prob(state)
        for sessions in ([], [3], [3, 10], [12, 15]):
            _, recalls = simulate(prob, sessions)
            for r in recalls:
                assert r is not None
                assert math.isfinite(r) and 0.0 <= r <= 1.0
        R = st.readiness(state, today=10)
        for info in R.values():
            if info["readiness"] is not None:
                assert math.isfinite(info["readiness"])
                assert 0.0 <= info["readiness"] <= 1.0


def test_stage3_studyforge_01_no_lost_bindings():
    state = _state_a()
    prob = _prob(state)
    assert ("Midterm", 0.6, 7) in prob.bindings
    assert ("Final", 0.5, 21) in prob.bindings
    data = dash.plan_data(_shared_config())
    assert len(data["topic_info"]["probability"]["bindings"]) == 2
    assert set(data["exam_summary"]) == {"Midterm", "Final"}
