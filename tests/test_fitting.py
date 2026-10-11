import json
import math
import os

import pytest

from studyforge import cli
from studyforge import state as st
from studyforge.memory import (
    MIN_FIT_SAMPLES,
    S_MAX,
    S_MIN,
    TopicState,
    fit_stability,
    initial_stability,
)

EXAMS = os.path.join(os.path.dirname(__file__), "..", "exams.yaml")
S_TRUE = 2.0


def run(args):
    cli.main(args)


def synthetic_pairs(S, intervals):
    return [(t, math.exp(-t / S)) for t in intervals]


def test_fit_recovers_known_stability():
    pairs = synthetic_pairs(2.5, [1, 2, 3, 5, 8, 13])
    assert fit_stability(pairs) == pytest.approx(2.5)


def test_fit_needs_minimum_samples():
    assert fit_stability(synthetic_pairs(2.0, [1, 2])) is None
    assert fit_stability(synthetic_pairs(2.0, [1, 2, 3])) == pytest.approx(2.0)
    assert MIN_FIT_SAMPLES == 3


def test_fit_skips_invalid_pairs():
    pairs = [
        (None, 0.5),   # first review: no interval
        (2.0, None),   # logged without a recall score
        (0, 0.8),      # same-day repeat: no information
        (-1, 0.5),     # out-of-order log: unusable
    ]
    assert fit_stability(pairs) is None
    pairs += synthetic_pairs(2.0, [2, 3, 4])
    assert fit_stability(pairs) == pytest.approx(2.0)


def test_fit_clamps_perfect_memory_to_max():
    pairs = [(t, 1.0) for t in [2, 4, 7]]
    assert fit_stability(pairs) == pytest.approx(S_MAX)


def test_fit_clamps_extreme_forgetting_to_min():
    pairs = [(0.5, 0.001) for _ in range(3)]
    assert fit_stability(pairs) == pytest.approx(S_MIN)


def test_fit_handles_zero_recall_without_blowing_up():
    pairs = [(2.0, 0.0), (3.0, 0.0), (4.0, 0.0)]
    S = fit_stability(pairs)
    assert S is not None
    assert S_MIN <= S <= S_MAX


def test_review_accepts_observed_recall():
    ts = TopicState("x", difficulty=3)
    s0 = ts.S
    R = ts.review(0, R=0.5)
    assert R == 0.5
    assert ts.S == pytest.approx(s0 * 1.5)


def test_log_with_recall_fits_personal_stability(tmp_path, capsys):
    sp = str(tmp_path / "state.json")
    run(["--state", sp, "plan", "--exams", EXAMS])
    # Four reviews; the last three give (interval, recall) pairs for S=2.0.
    run(["--state", sp, "log", "--topic", "Rotation", "--minutes", "25",
         "--day", "0", "--recall", "0.0"])
    for day, prev in [(2, 0), (5, 2), (9, 5)]:
        r = math.exp(-(day - prev) / S_TRUE)
        run(["--state", sp, "log", "--topic", "Rotation", "--minutes", "25",
             "--day", str(day), "--recall", f"{r:.4f}"])
    topics = st.load_state(sp)["topics"]
    assert topics["Rotation"]["S"] == pytest.approx(S_TRUE, rel=1e-3)
    assert topics["Rotation"]["S_source"] == "fitted"
    out = capsys.readouterr().out
    assert "Fitted your personal stability" in out


def test_log_stores_interval_and_recall(tmp_path):
    sp = str(tmp_path / "state.json")
    run(["--state", sp, "plan", "--exams", EXAMS])
    run(["--state", sp, "log", "--topic", "Rotation", "--minutes", "25",
         "--day", "0"])
    run(["--state", sp, "log", "--topic", "Rotation", "--minutes", "25",
         "--day", "3", "--recall", "0.4"])
    hist = st.load_state(sp)["history"]
    assert hist[0]["interval"] is None
    assert hist[0]["recall"] is None
    assert hist[1]["interval"] == 3
    assert hist[1]["recall"] == pytest.approx(0.4)


def test_log_without_recall_keeps_default_source(tmp_path):
    sp = str(tmp_path / "state.json")
    run(["--state", sp, "plan", "--exams", EXAMS])
    for day in [0, 2, 5, 9]:
        run(["--state", sp, "log", "--topic", "Rotation", "--minutes", "25",
             "--day", str(day)])
    topics = st.load_state(sp)["topics"]
    assert topics["Rotation"]["S_source"] == "default"


def test_log_rejects_bad_recall(tmp_path):
    sp = str(tmp_path / "state.json")
    run(["--state", sp, "plan", "--exams", EXAMS])
    with pytest.raises(SystemExit):
        run(["--state", sp, "log", "--topic", "Rotation", "--minutes", "25",
             "--recall", "1.5"])
    with pytest.raises(SystemExit):
        run(["--state", sp, "log", "--topic", "Rotation", "--minutes", "25",
             "--recall", "-0.1"])


def test_too_few_recall_points_falls_back(tmp_path):
    sp = str(tmp_path / "state.json")
    run(["--state", sp, "plan", "--exams", EXAMS])
    run(["--state", sp, "log", "--topic", "Rotation", "--minutes", "25",
         "--day", "0", "--recall", "0.0"])
    run(["--state", sp, "log", "--topic", "Rotation", "--minutes", "25",
         "--day", "2", "--recall", "0.37"])
    topics = st.load_state(sp)["topics"]
    assert topics["Rotation"]["S_source"] == "default"


def test_status_marks_fitted_topics(tmp_path, capsys):
    sp = str(tmp_path / "state.json")
    run(["--state", sp, "plan", "--exams", EXAMS])
    run(["--state", sp, "log", "--topic", "Rotation", "--minutes", "25",
         "--day", "0", "--recall", "0.0"])
    for day, prev in [(2, 0), (5, 2), (9, 5)]:
        r = math.exp(-(day - prev) / S_TRUE)
        run(["--state", sp, "log", "--topic", "Rotation", "--minutes", "25",
             "--day", str(day), "--recall", f"{r:.4f}"])
    run(["--state", sp, "status", "--today", "9"])
    out = capsys.readouterr().out
    lines = [ln for ln in out.splitlines() if ln.startswith("Rotation")]
    assert len(lines) == 1
    assert "fitted" in lines[0]
    assert "1/15 topics fitted" in out


def test_planner_uses_fitted_stability(tmp_path):
    sp = str(tmp_path / "state.json")
    run(["--state", sp, "plan", "--exams", EXAMS])
    state = st.load_state(sp)
    before = {t.name: t.S for t in cli._plan_topics(state)}
    assert before["Rotation"] == pytest.approx(initial_stability(4))
    run(["--state", sp, "log", "--topic", "Rotation", "--minutes", "25",
         "--day", "0", "--recall", "0.0"])
    for day, prev in [(2, 0), (5, 2), (9, 5)]:
        r = math.exp(-(day - prev) / S_TRUE)
        run(["--state", sp, "log", "--topic", "Rotation", "--minutes", "25",
             "--day", str(day), "--recall", f"{r:.4f}"])
    state = st.load_state(sp)
    after = {t.name: t.S for t in cli._plan_topics(state)}
    assert after["Rotation"] == pytest.approx(S_TRUE, rel=1e-3)
