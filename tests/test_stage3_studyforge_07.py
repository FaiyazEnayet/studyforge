"""A07: explicit learning activities, retrieval success, feedback, transfer.

The defect: a generic 25-minute review is modeled as an automatic
stability increase even if recall failed or the task was first
learning, rereading, retrieval, problem solving, or corrective
feedback. Logging a review with observed recall 0 doubles stability,
so a failed retrieval counts as the strongest possible learning
event. These tests pin the fix: typed activities with distinct
acquisition, retrieval, transfer, and feedback labels; a failed
retrieval changes nothing; a missing outcome stays unknown; replay
is deterministic; and an ablation shows whether the added structure
predicts held-out outcomes better than the old update.
"""

import math
import os

import pytest

from studyforge import cli
from studyforge import state as st
from studyforge.activities import (
    ACTIVITY_KINDS,
    Activity,
    ablation,
    apply_activity,
    label,
    replay_activities,
)
from studyforge.memory import TopicState, initial_stability
from studyforge.models import ExponentialModel, ModelState
from studyforge.models.fsrs_adapter import (
    FSRSAdapter,
    to_studyforge_stability,
)
from studyforge.planner import PlanTopic, simulate

EXAMS = os.path.join(os.path.dirname(__file__), "..", "exams.yaml")


def _topic(difficulty=3):
    return TopicState("algebra", difficulty=difficulty)


def _failure_heavy_log():
    """Ten logged events where failures recur: the old update doubles
    stability on each one, the activity-aware update does not."""
    return [
        Activity("learn", 0, 25),
        Activity("retrieval", 1, 25, correctness=0.8),
        Activity("retrieval", 2, 25, correctness=0.0),
        Activity("retrieval", 3, 25, correctness=0.5),
        Activity("retrieval", 5, 25, correctness=0.0),
        Activity("retrieval", 6, 25, correctness=0.6),
        Activity("retrieval", 9, 25, correctness=0.0),
        Activity("retrieval", 10, 25, correctness=0.7),
        Activity("retrieval", 14, 25, correctness=0.0),
        Activity("retrieval", 15, 25, correctness=0.5),
    ]


def test_stage3_studyforge_07_failed_retrieval_is_not_learning():
    ts = _topic()
    ts.review(0)  # first learning: S 1.0 -> 2.0
    s_before = ts.S
    R = ts.review(4, R=0.0)
    assert R == 0.0
    assert ts.S == pytest.approx(s_before)
    assert ts.last_review == 0
    # A later successful retrieval still counts.
    ts.review(5, R=0.5)
    assert ts.S == pytest.approx(s_before * 1.5)
    assert ts.last_review == 5


def test_stage3_studyforge_07_first_review_still_counts_as_learning():
    # The gate is on an OBSERVED zero, not a predicted one: a first
    # review with no observed recall is acquisition, not a failure.
    ts = _topic()
    s0 = ts.S
    R = ts.review(4)
    assert R == 0.0
    assert ts.S == pytest.approx(s0 * 2.0)
    assert ts.last_review == 4


def test_stage3_studyforge_07_model_transition_failure_is_not_learning():
    model = ExponentialModel()
    state = ModelState(S=2.0, last_review=0, difficulty=3)
    after = model.transition(state, 4, outcome=0.0)
    assert after.S == pytest.approx(2.0)
    assert after.last_review == 0
    # A predicted zero (no outcome logged) keeps the old planning rule.
    planned = model.transition(state, 4, outcome=None)
    assert planned.S > 2.0
    assert planned.last_review == 4


def test_stage3_studyforge_07_fsrs_failure_is_not_learning():
    adapter = FSRSAdapter()
    state = ModelState(S=1.0, last_review=0, difficulty=3)
    after = adapter.transition(state, 4, outcome=0)
    assert after.last_review == 0
    # Stability is unchanged in FSRS units (the gate fires before the
    # StudyForge-unit conversion runs).
    assert after.S == pytest.approx(1.0)
    assert to_studyforge_stability(after.S) == pytest.approx(
        to_studyforge_stability(1.0))


def test_stage3_studyforge_07_labels_do_not_conflate():
    assert label(Activity("learn", 0, 25)) == "acquisition"
    assert label(Activity("worked-example", 0, 25)) == "acquisition"
    assert label(Activity("retrieval", 1, 25, correctness=1.0)) == "retrieval"
    assert label(Activity("explain", 1, 25, correctness=1.0)) == "retrieval"
    assert label(Activity("practice-problem", 1, 25, correctness=1.0,
                          item="repeated")) == "retrieval"
    # A novel item is a transfer test, whatever the retrieval kind.
    assert label(Activity("practice-problem", 1, 25, correctness=1.0,
                          item="novel")) == "transfer"
    assert label(Activity("retrieval", 1, 25, correctness=1.0,
                          item="novel")) == "transfer"
    assert label(Activity("feedback", 1, 10)) == "feedback"


def test_stage3_studyforge_07_missing_outcome_stays_unknown():
    ts = _topic()
    ts.review(0)
    s_before, lr_before = ts.S, ts.last_review
    result = apply_activity(
        ts, Activity("retrieval", 3, 25, correctness=None))
    assert result.label == "retrieval"
    assert result.outcome == "unknown"
    assert ts.S == pytest.approx(s_before)
    assert ts.last_review == lr_before


def test_stage3_studyforge_07_failed_retrieval_activity_records_failure():
    ts = _topic()
    ts.review(0)
    result = apply_activity(
        ts, Activity("retrieval", 4, 25, correctness=0.0))
    assert result.label == "retrieval"
    assert result.outcome == "failure"
    assert ts.S == pytest.approx(2.0)
    assert ts.last_review == 0


def test_stage3_studyforge_07_successful_retrieval_grows_stability():
    ts = _topic()
    ts.review(0)  # S 1.0 -> 2.0, last_review 0
    result = apply_activity(
        ts, Activity("retrieval", 4, 25, correctness=0.5))
    assert result.outcome == "success"
    assert ts.S == pytest.approx(2.0 * 1.5)
    assert ts.last_review == 4


def test_stage3_studyforge_07_learn_establishes_without_doubling():
    ts = _topic()
    s0 = ts.S
    result = apply_activity(ts, Activity("learn", 0, 25))
    assert result.label == "acquisition"
    assert result.outcome == "established"
    assert ts.S == pytest.approx(s0)
    assert ts.last_review == 0
    # Rereading later claims nothing.
    result2 = apply_activity(ts, Activity("learn", 2, 25))
    assert result2.outcome == "recorded"
    assert ts.S == pytest.approx(s0)
    assert ts.last_review == 0


def test_stage3_studyforge_07_feedback_alone_is_not_learning():
    ts = _topic()
    ts.review(0)
    result = apply_activity(ts, Activity("feedback", 1, 10))
    assert result.label == "feedback"
    assert result.outcome == "recorded"
    assert ts.S == pytest.approx(2.0)
    assert ts.last_review == 0


def test_stage3_studyforge_07_same_day_repeat_is_recorded_only():
    ts = _topic()
    ts.review(0)
    apply_activity(ts, Activity("retrieval", 2, 25, correctness=0.5))
    s_after = ts.S
    result = apply_activity(
        ts, Activity("retrieval", 2, 25, correctness=1.0))
    assert result.outcome == "recorded"
    assert ts.S == pytest.approx(s_after)
    assert ts.last_review == 2


def test_stage3_studyforge_07_activity_before_last_review_rejected():
    ts = _topic()
    ts.review(5)
    with pytest.raises(ValueError):
        apply_activity(ts, Activity("retrieval", 3, 25, correctness=0.5))


def test_stage3_studyforge_07_activity_validation():
    with pytest.raises(ValueError):
        Activity("nap", 0, 25)
    with pytest.raises(ValueError):
        Activity("retrieval", 0, 25, correctness=1.5)
    with pytest.raises(ValueError):
        Activity("retrieval", 0, 25, correctness=-0.1)
    with pytest.raises(ValueError):
        Activity("retrieval", 0, 25, correctness="high")
    with pytest.raises(ValueError):
        Activity("retrieval", 0, 25, correctness=True)
    with pytest.raises(ValueError):
        Activity("retrieval", 0, 0)
    with pytest.raises(ValueError):
        Activity("retrieval", -1, 25)
    with pytest.raises(ValueError):
        Activity("retrieval", 0, 25, item="surprise")
    assert ACTIVITY_KINDS == (
        "learn", "retrieval", "worked-example", "practice-problem",
        "explain", "feedback",
    )


def test_stage3_studyforge_07_replay_is_deterministic():
    events = _failure_heavy_log()
    first = replay_activities("algebra", 3, events)
    second = replay_activities("algebra", 3, events)
    assert first.S == pytest.approx(second.S)
    assert first.last_review == second.last_review
    # Input order does not matter: replay sorts by day.
    shuffled = [events[5], events[0], events[9], events[2], events[1],
                events[8], events[3], events[7], events[4], events[6]]
    third = replay_activities("algebra", 3, shuffled)
    assert third.S == pytest.approx(first.S)
    assert third.last_review == first.last_review


def test_stage3_studyforge_07_ablation_activity_aware_beats_baseline():
    result = ablation(_failure_heavy_log(), difficulty=3)
    assert result["n_train"] == 8
    assert result["n_holdout"] == 2
    assert result["n_scored"] == 2
    assert result["verdict"] == "activity-aware"
    assert result["activity_mae"] < result["baseline_mae"]
    assert result["activity_mae"] == pytest.approx(0.28896, rel=1e-3)
    assert result["baseline_mae"] == pytest.approx(0.70850, rel=1e-3)


def test_stage3_studyforge_07_ablation_agrees_without_failures():
    events = [
        Activity("retrieval", 0, 25, correctness=0.8),
        Activity("retrieval", 1, 25, correctness=0.5),
        Activity("retrieval", 3, 25, correctness=0.6),
        Activity("retrieval", 6, 25, correctness=0.7),
        Activity("retrieval", 10, 25, correctness=0.5),
        Activity("retrieval", 15, 25, correctness=0.8),
    ]
    result = ablation(events, difficulty=3)
    assert result["verdict"] == "tie"
    assert result["activity_mae"] == pytest.approx(result["baseline_mae"])


def test_stage3_studyforge_07_ablation_insufficient_data_is_honest():
    few = [
        Activity("retrieval", 0, 25, correctness=0.8),
        Activity("retrieval", 1, 25, correctness=0.5),
    ]
    result = ablation(few, difficulty=3)
    assert result["verdict"] == "insufficient data"
    assert result["baseline_mae"] is None
    assert result["activity_mae"] is None
    # Nothing scorable in the holdout: no numbers are invented.
    unscored = _failure_heavy_log()[:-2] + [
        Activity("retrieval", 14, 25, correctness=None),
        Activity("retrieval", 15, 25, correctness=None),
    ]
    result2 = ablation(unscored, difficulty=3)
    assert result2["verdict"] == "insufficient data"
    assert result2["n_scored"] == 0


def test_stage3_studyforge_07_ablation_is_deterministic():
    events = _failure_heavy_log()
    assert ablation(events, difficulty=3) == ablation(events, difficulty=3)


def test_stage3_studyforge_07_asof_replay_ignores_failed_retrieval():
    # History: day 0 recall 0.9, day 5 recall 0.0 (failed retrieval),
    # day 7 recall 0.8. The stored state (S 1.32, last review day 7)
    # already includes the day-7 review, so as of exam day 6 the
    # state must be rebuilt by replaying only the earlier events.
    # The failed event must not inflate stability: S stays 1.1 from
    # the day-0 success, last review day 0.
    topic = PlanTopic("algebra", [("Midterm", 1.0, 6)], 1.32, 7,
                      difficulty=3,
                      history=[(0, 0.9), (5, 0.0), (7, 0.8)])
    _, recalls = simulate(topic, [])
    assert recalls[0] == pytest.approx(math.exp(-6 / 1.1), rel=1e-6)


def test_stage3_studyforge_07_log_failed_retrieval_end_to_end(
        tmp_path, capsys):
    sp = str(tmp_path / "state.json")
    cli.main(["--state", sp, "plan", "--exams", EXAMS])
    before = st.load_state(sp)["topics"]["rotation"]
    cli.main(["--state", sp, "log", "--topic", "Rotation", "--minutes", "25",
              "--day", "0", "--recall", "0"])
    after = st.load_state(sp)["topics"]["rotation"]
    assert after["last_review"] is None
    assert after["S"] == pytest.approx(
        initial_stability(before["difficulty"]))
    history = st.load_state(sp)["history"]
    assert len(history) == 1
    assert history[0]["recall"] == 0.0
    assert history[0]["activity"] == "retrieval"
    assert "Failed retrieval" in capsys.readouterr().out


def test_stage3_studyforge_07_failed_first_log_keeps_fit_intervals(tmp_path):
    # A failed first retrieval does not stamp last_review, but the
    # interval of the next log is still measured from it: the pair
    # stays usable for fitting, so the known S=2.0 decay is recovered.
    import math as _math

    sp = str(tmp_path / "state.json")
    cli.main(["--state", sp, "plan", "--exams", EXAMS])
    cli.main(["--state", sp, "log", "--topic", "Rotation", "--minutes", "25",
              "--day", "0", "--recall", "0"])
    for day, prev in [(2, 0), (5, 2), (9, 5)]:
        r = _math.exp(-(day - prev) / 2.0)
        cli.main(["--state", sp, "log", "--topic", "Rotation",
                  "--minutes", "25", "--day", str(day),
                  "--recall", f"{r:.4f}"])
    history = st.load_state(sp)["history"]
    assert [h["interval"] for h in history] == [None, 2, 3, 4]
    topics = st.load_state(sp)["topics"]
    assert topics["rotation"]["S"] == pytest.approx(2.0, rel=1e-3)
    assert topics["rotation"]["S_source"] == "fitted"


def test_stage3_studyforge_07_log_records_typed_activity(tmp_path):
    sp = str(tmp_path / "state.json")
    cli.main(["--state", sp, "plan", "--exams", EXAMS])
    cli.main(["--state", sp, "log", "--topic", "Rotation", "--minutes", "25",
              "--day", "0", "--activity", "practice-problem",
              "--item", "novel", "--recall", "0.8", "--feedback"])
    history = st.load_state(sp)["history"]
    assert len(history) == 1
    entry = history[0]
    assert entry["activity"] == "practice-problem"
    assert entry["item"] == "novel"
    assert entry["feedback"] is True
    assert entry["recall"] == 0.8
    # A successful novel practice problem is transfer: it counts.
    after = st.load_state(sp)["topics"]["rotation"]
    assert after["last_review"] == 0
    assert after["S"] > initial_stability(after["difficulty"])


def test_stage3_studyforge_07_log_missing_recall_stays_unknown(tmp_path):
    sp = str(tmp_path / "state.json")
    cli.main(["--state", sp, "plan", "--exams", EXAMS])
    cli.main(["--state", sp, "log", "--topic", "Rotation", "--minutes", "25",
              "--day", "0"])
    after = st.load_state(sp)["topics"]["rotation"]
    # First session is acquisition: the topic is established, and the
    # default stability stands because nothing was observed.
    assert after["last_review"] == 0
    assert after["S"] == pytest.approx(
        initial_stability(after["difficulty"]))
    history = st.load_state(sp)["history"]
    assert history[0]["recall"] is None
    assert history[0]["activity"] == "learn"
    # A later retrieval with no recorded outcome claims nothing.
    cli.main(["--state", sp, "log", "--topic", "Rotation", "--minutes", "25",
              "--day", "2"])
    after2 = st.load_state(sp)["topics"]["rotation"]
    assert after2["last_review"] == 0
    assert after2["S"] == pytest.approx(
        initial_stability(after2["difficulty"]))
    history2 = st.load_state(sp)["history"]
    assert history2[1]["activity"] == "retrieval"
    assert history2[1]["recall"] is None


def _state_with_history(tmp_path, entry):
    cfg = {
        "exams": [
            {"name": "Midterm", "day": 7, "topics": [
                {"name": "Algebra", "difficulty": 3, "weight": 1.0},
            ]},
        ],
        "rest_days": [],
        "daily_budget_minutes": 60,
        "session_minutes": 25,
    }
    path = str(tmp_path / "state.json")
    state = st.new_state(cfg)
    state["history"].append(entry)
    st.save_state(state, path)
    return path


def test_stage3_studyforge_07_state_rejects_bad_activity_fields(tmp_path):
    base = {"day": 2, "topic": "algebra", "minutes": 25,
            "interval": None, "recall": 0.9}
    for bad in (
        {"activity": "nap"},
        {"item": "surprise"},
        {"assistance": "yes"},
        {"feedback": 1},
    ):
        entry = dict(base, **bad)
        with pytest.raises(ValueError):
            st.load_state(_state_with_history(tmp_path, entry))


def test_stage3_studyforge_07_state_accepts_typed_history(tmp_path):
    entry = {"day": 2, "topic": "algebra", "minutes": 25,
             "interval": None, "recall": 0.9, "activity": "retrieval",
             "item": "novel", "assistance": False, "feedback": True}
    loaded = st.load_state(_state_with_history(tmp_path, entry))
    assert loaded["history"][0]["activity"] == "retrieval"
    assert loaded["history"][0]["item"] == "novel"
