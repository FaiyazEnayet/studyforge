"""ST1: stable topic IDs separate knowledge from exam bindings."""

import pytest

from studyforge import cli as _cli
from studyforge import state as st
from studyforge.planner import PlanTopic, objective, simulate


def _shared_probability_config():
    return {
        "exams": [
            {"name": "Midterm", "day": 7, "topics": [
                {"name": "Probability", "difficulty": 3, "weight": 0.6},
                {"name": "Algebra", "difficulty": 2, "weight": 0.4},
            ]},
            {"name": "Final", "day": 21, "topics": [
                {"name": "Probability", "difficulty": 3, "weight": 0.5},
                {"name": "Calculus", "difficulty": 4, "weight": 0.5},
            ]},
        ],
        "rest_days": [],
        "daily_budget_minutes": 60,
        "session_minutes": 25,
    }


def test_shared_topic_keeps_one_memory_state_and_two_bindings():
    state = st.new_state(_shared_probability_config())
    # one knowledge state for Probability, not two
    assert list(state["topics"]) == ["probability", "algebra", "calculus"]
    prob = state["topics"]["probability"]
    assert prob["name"] == "Probability"
    assert prob["S"] is None and prob["last_review"] is None
    # both exams bind the same topic ID with their own weights
    mid = next(e for e in state["config"]["exams"] if e["name"] == "Midterm")
    fin = next(e for e in state["config"]["exams"] if e["name"] == "Final")
    assert {"id": "probability", "weight": 0.6} in mid["topics"]
    assert {"id": "probability", "weight": 0.5} in fin["topics"]


def test_shared_topic_contributes_to_both_exam_objectives():
    state = st.new_state(_shared_probability_config())
    topics = _cli._plan_topics(state)
    prob = next(t for t in topics if t.name == "probability")
    # both bindings present with their own deadlines
    assert ("Midterm", 0.6, 7) in prob.bindings
    assert ("Final", 0.5, 21) in prob.bindings
    # a review before the midterm lifts both recalls, but the midterm's
    # recall ignores sessions scheduled after its deadline
    _, (r_mid, r_fin) = simulate(prob, [3])
    _, (r_mid_late, r_fin_late) = simulate(prob, [3, 10])
    assert r_mid_late == pytest.approx(r_mid)  # day-10 review is post-midterm
    assert r_fin_late > r_fin  # but it helps the final
    # objective counts both bindings
    obj = objective([prob], {"probability": [3]})
    assert obj == pytest.approx(0.6 * r_mid + 0.5 * r_fin)


def test_shared_topic_single_memory_state_in_readiness():
    state = st.new_state(_shared_probability_config())
    # log a review once: both exams see the same updated memory
    state["topics"]["probability"]["S"] = 5.0
    state["topics"]["probability"]["last_review"] = 2
    R = st.readiness(state, today=2)
    assert set(R) == {"Midterm", "Final"}
    # both exams include Probability; readiness differs by deadline/weight
    assert R["Midterm"]["readiness"] != R["Final"]["readiness"]
    assert R["Midterm"]["topics"] == 2 and R["Final"]["topics"] == 2


def test_duplicate_name_within_exam_rejected():
    cfg = {
        "exams": [{"name": "Midterm", "day": 7, "topics": [
            {"name": "Probability", "difficulty": 3, "weight": 0.6},
            {"name": "probability", "difficulty": 2, "weight": 0.4},
        ]}],
        "rest_days": [],
        "daily_budget_minutes": 60,
        "session_minutes": 25,
    }
    with pytest.raises(ValueError, match="duplicate topic"):
        st.new_state(cfg)


def test_topic_id_stable_and_slugified():
    assert st.topic_id("Probability") == "probability"
    assert st.topic_id("Bayes' Theorem!") == "bayes-theorem"
    assert st.topic_id("Probability") == st.topic_id("probability")
    with pytest.raises(ValueError):
        st.topic_id("!!!")
