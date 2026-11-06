"""Stage 3 studyforge ST2: validate scheduler inputs and persisted state.

Failing-first regression tests for the defects on main (HEAD 6b4e2f8):
load_config and new_state accept zero or negative session lengths and
daily caps, nonfinite weights, duplicate exam names, empty exam sets
and topic names that slug to the same ID without complaint; PlanTopic
and build_plan accept nonfinite stability, empty bindings and
non-positive scheduling parameters; load_state trusts persisted
bindings, stability values and history entries it never checks. Every
rejection below must name the bad value and say what to fix.
"""

import json
import math

import pytest

from studyforge import state as st
from studyforge.planner import PlanTopic, build_plan, to_schedule


def _config(**overrides):
    cfg = {
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
    cfg.update(overrides)
    return cfg


def _write_yaml(tmp_path, text, name="exams.yaml"):
    path = tmp_path / name
    path.write_text(text)
    return str(path)


def _write_state(tmp_path, state, name="state.json"):
    path = tmp_path / name
    with open(path, "w") as f:
        json.dump(state, f, indent=1)
    return str(path)


def _valid_state():
    return st.new_state(_config())


# --- config file loading -------------------------------------------------

def test_zero_session_minutes_rejected(tmp_path):
    path = _write_yaml(tmp_path, "session_minutes: 0\nexams:\n"
                       "  - name: Midterm\n    day: 7\n"
                       "    topics: [{name: Algebra, weight: 1.0}]\n")
    with pytest.raises(ValueError, match="session_minutes.*0"):
        st.load_config(path)


def test_negative_daily_cap_rejected(tmp_path):
    path = _write_yaml(tmp_path, "daily_budget_minutes: -30\n"
                       "session_minutes: 25\nexams:\n"
                       "  - name: Midterm\n    day: 7\n"
                       "    topics: [{name: Algebra, weight: 1.0}]\n")
    with pytest.raises(ValueError, match="daily_budget_minutes.*-30"):
        st.load_config(path)


def test_duplicate_exam_names_rejected(tmp_path):
    path = _write_yaml(tmp_path, "exams:\n"
                       "  - name: Midterm\n    day: 7\n"
                       "    topics: [{name: Algebra, weight: 1.0}]\n"
                       "  - name: Midterm\n    day: 14\n"
                       "    topics: [{name: Calculus, weight: 1.0}]\n")
    with pytest.raises(ValueError, match="duplicate exam.*Midterm"):
        st.load_config(path)


def test_empty_exam_set_rejected(tmp_path):
    path = _write_yaml(tmp_path, "session_minutes: 25\nexams: []\n")
    with pytest.raises(ValueError, match="at least one exam"):
        st.load_config(path)


def test_nonfinite_weight_rejected(tmp_path):
    path = _write_yaml(tmp_path, "exams:\n"
                       "  - name: Midterm\n    day: 7\n"
                       "    topics: [{name: Algebra, weight: .inf}]\n")
    with pytest.raises(ValueError, match="weight.*finite"):
        st.load_config(path)


def test_difficulty_out_of_range_rejected(tmp_path):
    path = _write_yaml(tmp_path, "exams:\n"
                       "  - name: Midterm\n    day: 7\n"
                       "    topics: [{name: Algebra, difficulty: 9,"
                       " weight: 1.0}]\n")
    with pytest.raises(ValueError, match="difficulty.*1.*5"):
        st.load_config(path)


def test_exam_missing_day_names_the_exam(tmp_path):
    path = _write_yaml(tmp_path, "exams:\n"
                       "  - name: Midterm\n"
                       "    topics: [{name: Algebra, weight: 1.0}]\n")
    with pytest.raises(ValueError, match="Midterm.*day"):
        st.load_config(path)


def test_empty_yaml_file_rejected(tmp_path):
    path = _write_yaml(tmp_path, "")
    with pytest.raises(ValueError, match="exams"):
        st.load_config(path)


# --- new_state validation ------------------------------------------------

def test_slug_collision_across_exams_rejected():
    cfg = _config()
    cfg["exams"][1]["topics"][0] = {
        "name": "Bayes Theorem", "difficulty": 3, "weight": 0.5}
    cfg["exams"][0]["topics"][0] = {
        "name": "Bayes' Theorem", "difficulty": 3, "weight": 0.6}
    with pytest.raises(ValueError, match="slug|same ID"):
        st.new_state(cfg)


def test_shared_topic_same_name_still_allowed():
    state = st.new_state(_config())
    assert list(state["topics"]) == ["probability", "algebra", "calculus"]


def test_duplicate_exam_names_rejected_in_new_state():
    cfg = _config()
    cfg["exams"].append(dict(cfg["exams"][0]))
    with pytest.raises(ValueError, match="duplicate exam.*Midterm"):
        st.new_state(cfg)


def test_exam_with_no_topics_rejected():
    cfg = _config(exams=[{"name": "Midterm", "day": 7, "topics": []}])
    with pytest.raises(ValueError, match="at least one topic"):
        st.new_state(cfg)


def test_negative_exam_day_rejected():
    cfg = _config()
    cfg["exams"][0]["day"] = -3
    with pytest.raises(ValueError, match="day.*-3"):
        st.new_state(cfg)


def test_malformed_day0_rejected():
    state = _valid_state()
    state["day0"] = "not-a-date"
    with pytest.raises(ValueError, match="day0.*YYYY-MM-DD"):
        st.day0_today(state)


# --- persisted state validation ------------------------------------------

def test_binding_to_unknown_topic_rejected(tmp_path):
    state = _valid_state()
    state["config"]["exams"][0]["topics"].append(
        {"id": "no-such-topic", "weight": 0.5})
    with pytest.raises(ValueError, match="no-such-topic"):
        st.load_state(_write_state(tmp_path, state))


def test_topic_with_no_binding_rejected(tmp_path):
    state = _valid_state()
    state["topics"]["orphan"] = {
        "name": "Orphan", "difficulty": 3, "S": None,
        "S_source": "default", "last_review": None}
    with pytest.raises(ValueError, match="orphan"):
        st.load_state(_write_state(tmp_path, state))


def test_nonfinite_stability_in_state_rejected(tmp_path):
    state = _valid_state()
    state["topics"]["probability"]["S"] = math.inf
    with pytest.raises(ValueError, match="stability.*finite"):
        st.load_state(_write_state(tmp_path, state))


def test_history_recall_out_of_range_rejected(tmp_path):
    state = _valid_state()
    state["history"].append(
        {"day": 2, "topic": "probability", "minutes": 25,
         "interval": None, "recall": 1.5})
    with pytest.raises(ValueError, match="recall.*0.*1"):
        st.load_state(_write_state(tmp_path, state))


def test_history_review_before_previous_review_rejected(tmp_path):
    state = _valid_state()
    state["history"].extend([
        {"day": 5, "topic": "probability", "minutes": 25,
         "interval": None, "recall": 0.9},
        {"day": 3, "topic": "probability", "minutes": 25,
         "interval": 1, "recall": 0.8},
    ])
    with pytest.raises(ValueError, match="day 3.*day 5|before.*previous"):
        st.load_state(_write_state(tmp_path, state))


def test_history_negative_day_rejected(tmp_path):
    state = _valid_state()
    state["history"].append(
        {"day": -1, "topic": "probability", "minutes": 25,
         "interval": None, "recall": 0.9})
    with pytest.raises(ValueError, match="day.*-1"):
        st.load_state(_write_state(tmp_path, state))


# --- planner validation --------------------------------------------------

def test_plan_topic_rejects_nonfinite_stability():
    with pytest.raises(ValueError, match="stability.*finite"):
        PlanTopic("A", [("E", 0.5, 7)], S=math.inf, last_review=None)


def test_plan_topic_rejects_zero_stability():
    with pytest.raises(ValueError, match="stability.*0"):
        PlanTopic("A", [("E", 0.5, 7)], S=0.0, last_review=None)


def test_plan_topic_rejects_empty_bindings():
    with pytest.raises(ValueError, match="binding"):
        PlanTopic("A", [], S=1.5, last_review=None)


def test_plan_topic_rejects_duplicate_exam_binding():
    with pytest.raises(ValueError, match="duplicate.*E"):
        PlanTopic("A", [("E", 0.5, 7), ("E", 0.5, 14)],
                  S=1.5, last_review=None)


def test_plan_topic_rejects_bad_history_recall():
    with pytest.raises(ValueError, match="recall.*0.*1"):
        PlanTopic("A", [("E", 0.5, 7)], S=1.5, last_review=3,
                  difficulty=3, history=[(1, 1.2)])


def test_build_plan_rejects_zero_session_length():
    topics = [PlanTopic("A", [("E", 0.5, 7)], S=1.5, last_review=None)]
    with pytest.raises(ValueError, match="session_min.*0"):
        build_plan(topics, horizon=7, daily_cap=60, session_min=0,
                   rest_days=set())


def test_build_plan_rejects_negative_daily_cap():
    topics = [PlanTopic("A", [("E", 0.5, 7)], S=1.5, last_review=None)]
    with pytest.raises(ValueError, match="daily_cap.*-10"):
        build_plan(topics, horizon=7, daily_cap=-10, session_min=25,
                   rest_days=set())


def test_build_plan_rejects_empty_topics():
    with pytest.raises(ValueError, match="at least one topic"):
        build_plan([], horizon=7, daily_cap=60, session_min=25,
                   rest_days=set())


# --- backlog contract: elapsed exams, caps, exam-day rule ----------------

def test_elapsed_exam_is_planned_around_not_rejected():
    topics = [PlanTopic("A", [("E", 0.5, 7)], S=1.5, last_review=None),
              PlanTopic("B", [("F", 0.5, 21)], S=1.0, last_review=None)]
    plan, obj = build_plan(topics, horizon=21, daily_cap=60,
                           session_min=25, rest_days=set(), start_day=10)
    assert plan["A"] == []
    assert obj > 0.0  # only the future exam contributes, and B gets study
    assert all(d < 21 for d in plan["B"])


def test_plan_never_exceeds_cap_and_never_touches_exam_day():
    topics = [PlanTopic("A", [("E", 0.5, 7)], S=1.5, last_review=None),
              PlanTopic("B", [("E", 0.5, 7)], S=1.0, last_review=None)]
    plan, _ = build_plan(topics, horizon=7, daily_cap=50,
                         session_min=25, rest_days={3})
    sched = to_schedule(plan, 25)
    for day, sessions in sched.items():
        assert sum(m for _, m in sessions) <= 50
        assert day < 7
        assert day != 3
