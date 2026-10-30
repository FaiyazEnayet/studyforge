"""ST1 foundation: persisted schema version and legacy state migration.

The legacy fixture below replicates, field for field, the state built
by new_state before commit 85e175b ("Separate stable topic IDs from
exam bindings") — see `git show 85e175b^:src/studyforge/state.py`:
topics keyed by display name, each carrying its exam, exam day and
weight inline (so a topic shared by two exams kept only the last
exam's association), config exams holding {name, difficulty, weight}
topic dicts, and no schema_version anywhere.
"""

import json

import pytest

from studyforge import state as st

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
                "S": None,  # filled from difficulty on first use
                "S_source": "default",  # or "fitted" once logs personalize it
                "last_review": None,
            }
    return {
        "day0": day0,
        "config": config,
        "topics": topics,
        "history": [],
        "plan": {},
    }


def _used_legacy_state():
    """A legacy state after a logged review of the shared topic."""
    state = _legacy_state(_config())
    prob = state["topics"]["Probability"]
    prob["S"] = 5.0
    prob["S_source"] = "fitted"
    prob["last_review"] = 2
    # history entry in the pre-85e175b cli shape (topic = display name)
    state["history"].append(
        {"day": 2, "topic": "Probability", "minutes": 25,
         "interval": None, "recall": 0.9}
    )
    return state


def _write(tmp_path, state, name="state.json"):
    path = tmp_path / name
    with open(path, "w") as f:
        json.dump(state, f, indent=1)
    return str(path)


def test_legacy_fresh_state_migrates_to_new_state_shape(tmp_path):
    config = _config()
    path = _write(tmp_path, _legacy_state(config))
    migrated = st.load_state(path)
    expected = st.new_state(config)
    assert migrated["schema_version"] == st.SCHEMA_VERSION == 1
    # identical topics and config to what new_state produces today
    assert migrated["topics"] == expected["topics"]
    assert migrated["config"] == expected["config"]
    # everything else carries over untouched
    assert migrated["day0"] == DAY0
    assert migrated["history"] == [] and migrated["plan"] == {}


def test_legacy_shared_topic_keeps_memory_and_both_bindings(tmp_path):
    path = _write(tmp_path, _used_legacy_state())
    migrated = st.load_state(path)
    # one memory state for the shared topic, memory fields preserved
    assert list(migrated["topics"]) == ["probability", "algebra", "calculus"]
    prob = migrated["topics"]["probability"]
    assert prob == {
        "name": "Probability", "difficulty": 3,
        "S": 5.0, "S_source": "fitted", "last_review": 2,
    }
    # no lost bindings: both exams bind it, each with its own weight
    mid = next(e for e in migrated["config"]["exams"] if e["name"] == "Midterm")
    fin = next(e for e in migrated["config"]["exams"] if e["name"] == "Final")
    assert {"id": "probability", "weight": 0.6} in mid["topics"]
    assert {"id": "probability", "weight": 0.5} in fin["topics"]
    # history passes through unchanged, and the migrated state is
    # directly consumable by the current code
    assert migrated["history"][0]["topic"] == "Probability"
    R = st.readiness(migrated, today=2)
    assert set(R) == {"Midterm", "Final"}
    assert R["Midterm"]["topics"] == 2 and R["Final"]["topics"] == 2


def test_migration_is_idempotent(tmp_path):
    legacy = _used_legacy_state()
    once = st.migrate_state(legacy)
    twice = st.migrate_state(once)
    assert twice == once
    # a current-format file loads byte-for-byte unchanged in content
    path = _write(tmp_path, once)
    assert st.load_state(path) == once


def test_round_trip_serialization_is_stable(tmp_path):
    # current state: save -> load -> save reproduces the same bytes
    state = st.new_state(_config())
    p1 = tmp_path / "a.json"
    st.save_state(state, str(p1))
    st.save_state(st.load_state(str(p1)), str(tmp_path / "b.json"))
    assert (tmp_path / "b.json").read_bytes() == p1.read_bytes()
    # legacy state: after one migrate+save cycle the bytes stabilize
    legacy_path = _write(tmp_path, _used_legacy_state(), "legacy.json")
    p2 = tmp_path / "c.json"
    st.save_state(st.load_state(legacy_path), str(p2))
    st.save_state(st.load_state(str(p2)), str(tmp_path / "d.json"))
    assert (tmp_path / "d.json").read_bytes() == p2.read_bytes()
    assert json.loads(p2.read_text())["schema_version"] == 1


def test_current_format_loads_unchanged(tmp_path):
    state = st.new_state(_config())
    path = _write(tmp_path, state)
    loaded = st.load_state(path)
    assert loaded == state
    assert loaded["schema_version"] == 1


def test_corrupt_state_fails_and_leaves_file_untouched(tmp_path):
    path = tmp_path / "state.json"
    path.write_bytes(b"{not valid json")
    before = path.read_bytes()
    with pytest.raises(ValueError, match="not valid JSON"):
        st.load_state(str(path))
    assert path.read_bytes() == before


def test_unmigratable_legacy_state_fails_and_leaves_file_untouched(tmp_path):
    bad = _legacy_state(_config())
    bad["topics"] = "not-an-object"
    path = _write(tmp_path, bad)
    before = (tmp_path / "state.json").read_bytes()
    with pytest.raises(ValueError, match="cannot migrate"):
        st.load_state(path)
    assert (tmp_path / "state.json").read_bytes() == before


def test_future_schema_version_rejected(tmp_path):
    state = st.new_state(_config())
    state["schema_version"] = 99
    path = _write(tmp_path, state)
    before = (tmp_path / "state.json").read_bytes()
    with pytest.raises(ValueError, match="unsupported state schema_version"):
        st.load_state(path)
    assert (tmp_path / "state.json").read_bytes() == before
