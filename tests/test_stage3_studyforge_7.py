"""ST7: planner invariants and migration, documented and pinned.

Most of these invariants were already enforced by the ST1 foundation
and ST2 validation; this file pins the ones that had no dedicated
regression test (backup preservation on save, same-day planner
behavior, deterministic regeneration of the example plan) and
re-pins round-trip serialization and idempotent migration under the
ST7 name, so docs/planner-invariants.md can point at one file per
invariant. The legacy fixture below replicates, field for field, the
state built by new_state before commit 85e175b (see
tests/test_stage3_studyforge_st1.py for the full provenance note).
"""

import json
import os

import pytest

from studyforge import state as st
from studyforge.planner import PlanTopic, build_plan, simulate

DAY0 = "2026-10-18"


def _config():
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


def _legacy_state(config, day0=DAY0):
    """A version 0 state, built exactly as pre-85e175b new_state did."""
    topics = {}
    for ex in config["exams"]:
        for t in ex["topics"]:
            topics[t["name"]] = {
                "exam": ex["name"],
                "exam_day": ex["day"],
                "difficulty": t["difficulty"],
                "weight": t["weight"],
                "S": None,
                "S_source": "default",
                "last_review": None,
            }
    return {
        "day0": day0,
        "config": config,
        "topics": topics,
        "history": [],
        "plan": {},
    }


def _topic():
    """One topic, last reviewed on day 3 with stability 2.0."""
    return PlanTopic("algebra", [("Midterm", 1.0, 10)], 2.0, 3)


def _read(path):
    with open(path) as f:
        return json.load(f)


def _write(path, payload):
    with open(path, "w") as f:
        json.dump(payload, f, indent=1)


def test_save_preserves_backup_of_previous_state(tmp_path):
    path = str(tmp_path / "state.json")
    first = st.new_state(_config())
    st.save_state(first, path)
    assert not os.path.exists(path + ".bak")
    first_bytes = open(path, "rb").read()
    second = st.new_state(_config())
    second["topics"]["algebra"]["S"] = 4.0
    st.save_state(second, path)
    assert open(path + ".bak", "rb").read() == first_bytes
    assert _read(path)["topics"]["algebra"]["S"] == 4.0
    assert _read(path + ".bak")["topics"]["algebra"]["S"] is None


def test_backup_holds_only_the_immediately_previous_state(tmp_path):
    path = str(tmp_path / "state.json")
    for s_value in (None, 3.0, 5.0):
        state = st.new_state(_config())
        state["topics"]["algebra"]["S"] = s_value
        st.save_state(state, path)
    assert _read(path)["topics"]["algebra"]["S"] == 5.0
    assert _read(path + ".bak")["topics"]["algebra"]["S"] == 3.0


def test_saved_state_still_round_trips_with_backup_present(tmp_path):
    path = str(tmp_path / "state.json")
    state = st.new_state(_config())
    st.save_state(state, path)
    state["topics"]["algebra"]["S"] = 2.5
    st.save_state(state, path)
    loaded = st.load_state(path)
    assert loaded["schema_version"] == st.SCHEMA_VERSION
    assert loaded["topics"]["algebra"]["S"] == 2.5
    assert os.path.exists(path + ".bak")


def test_planner_session_on_last_review_day_is_a_noop():
    topic = _topic()
    s_no_session, recalls_no_session = simulate(topic, [])
    s_same_day, recalls_same_day = simulate(topic, [3])
    assert s_same_day == s_no_session
    assert recalls_same_day == recalls_no_session


def test_planner_second_same_day_session_is_a_noop():
    topic = _topic()
    s_one, recalls_one = simulate(topic, [5])
    s_two, recalls_two = simulate(topic, [5, 5])
    assert s_two == s_one
    assert recalls_two == recalls_one


def test_planner_next_day_session_counts():
    topic = _topic()
    s_none, _ = simulate(topic, [])
    s_next, _ = simulate(topic, [4])
    assert s_next > s_none


def test_example_plan_regenerates_deterministically():
    from studyforge.cli import _plan_topics

    root = os.path.join(os.path.dirname(__file__), "..")
    cfg = st.load_config(os.path.join(root, "exams.yaml"))
    plans = []
    objectives = []
    for _ in range(2):
        state = st.new_state(cfg)
        topics = _plan_topics(state)
        horizon = max(t.latest_exam_day for t in topics)
        plan, obj = build_plan(
            topics, horizon, cfg["daily_budget_minutes"],
            cfg["session_minutes"], set(cfg["rest_days"]),
            start_day=0, seed=0,
        )
        plans.append(plan)
        objectives.append(obj)
    assert plans[0] == plans[1]
    assert objectives[0] == objectives[1]
    assert sum(len(v) for v in plans[0].values()) == 72
    assert objectives[0] == pytest.approx(2.278051, rel=1e-4)


def test_round_trip_serialization_is_stable(tmp_path):
    path = str(tmp_path / "state.json")
    state = st.new_state(_config())
    state["topics"]["algebra"]["S"] = 3.5
    state["topics"]["algebra"]["last_review"] = 2
    state["history"].append(
        {"day": 2, "topic": "algebra", "minutes": 25,
         "interval": None, "recall": 0.9}
    )
    st.save_state(state, path)
    assert st.load_state(path) == _read(path)
    st.save_state(st.load_state(path), path)
    assert _read(path) == st.load_state(path)


def test_migration_is_idempotent(tmp_path):
    legacy = _legacy_state(_config())
    once = st.migrate_state(legacy)
    twice = st.migrate_state(once)
    assert once == twice
    assert once["schema_version"] == st.SCHEMA_VERSION
    assert legacy.get("schema_version") is None
    assert "Probability" not in once["topics"]
    assert once["topics"]["probability"]["name"] == "Probability"


def test_loading_legacy_state_leaves_file_untouched(tmp_path):
    path = str(tmp_path / "state.json")
    _write(path, _legacy_state(_config()))
    before = open(path, "rb").read()
    loaded = st.load_state(path)
    assert loaded["schema_version"] == st.SCHEMA_VERSION
    assert open(path, "rb").read() == before
