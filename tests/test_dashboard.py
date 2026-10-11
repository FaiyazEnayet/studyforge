"""Tests for studyforge.dashboard: the view-layer helpers.

All expected values were produced by running the real engine on the
included exams.yaml, never hand-written.
"""

import math
import os
import py_compile

import pytest
import yaml

from studyforge import dashboard as dash
from studyforge.planner import simulate


def load_example_config():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "exams.yaml")) as f:
        cfg = yaml.safe_load(f)
    cfg["rest_days"] = sorted(set(cfg.get("rest_days", [])))
    return cfg


def test_plan_data_matches_cli_numbers():
    data = dash.plan_data(load_example_config())
    assert data["n_sessions"] == 72
    assert data["objective"] == pytest.approx(2.278051, rel=1e-4)
    assert data["total_minutes"] == 1800
    assert len(data["topic_info"]) == 15


def test_plan_data_exam_summary():
    data = dash.plan_data(load_example_config())
    ex = data["exam_summary"]
    assert ex["Psychology Midterm"]["sessions"] == 13
    assert ex["Physics I Midterm"]["sessions"] == 21
    assert ex["Calculus III Final"]["sessions"] == 38
    assert ex["Psychology Midterm"]["recall_at_exam"] == pytest.approx(
        0.6795, rel=1e-3)
    assert ex["Physics I Midterm"]["recall_at_exam"] == pytest.approx(
        0.7759, rel=1e-3)
    assert ex["Calculus III Final"]["recall_at_exam"] == pytest.approx(
        0.8226, rel=1e-3)


def test_plan_data_respects_constraints():
    cfg = load_example_config()
    data = dash.plan_data(cfg)
    cap = cfg["daily_budget_minutes"]
    rest = set(cfg["rest_days"])
    for d, minutes in data["day_minutes"].items():
        assert minutes <= cap
        assert d not in rest
    # no session on or after its exam day
    for name, info in data["topic_info"].items():
        assert all(d < info["exam_day"] for d in info["days"])


def test_recall_trajectory_marks_match_plan_days():
    cfg = load_example_config()
    data = dash.plan_data(cfg)
    info = data["topic_info"]["Rotation"]
    traj = dash.recall_trajectory("Rotation", info["days"], info["S0"],
                                  info["exam_day"], data["horizon"])
    mark_days = [m[0] for m in traj["review_marks"]]
    assert mark_days == [float(d) for d in info["days"]]
    # marks sit at the pre-review recall (the dip), not at 100%
    for day, r in traj["review_marks"]:
        assert 0.0 <= r <= 1.0
    assert traj["review_marks"][0][1] == pytest.approx(0.0)


def test_recall_trajectory_matches_simulate():
    from studyforge.planner import PlanTopic

    cfg = load_example_config()
    data = dash.plan_data(cfg)
    info = data["topic_info"]["Rotation"]
    traj = dash.recall_trajectory("Rotation", info["days"], info["S0"],
                                  info["exam_day"], data["horizon"])
    tp = PlanTopic("Rotation", info["weight"], info["exam_day"],
                   info["S0"], None)
    S_sim, R_sim = simulate(tp, info["days"])
    assert traj["S_final"] == pytest.approx(S_sim)
    assert traj["recall_at_exam"] == pytest.approx(R_sim)
    assert traj["S_final"] == pytest.approx(8.2926, rel=1e-3)
    assert traj["recall_at_exam"] == pytest.approx(0.6964, rel=1e-3)


def test_recall_trajectory_curve_is_monotone_between_reviews():
    traj = dash.recall_trajectory("X", [3, 7], 1.0, 14, 14)
    xs, ys = traj["xs"], traj["ys"]
    # find segments between review days; recall must not rise within one
    bounds = [0.0] + [m[0] for m in traj["review_marks"]] + [14.0]
    for lo, hi in zip(bounds, bounds[1:]):
        seg = [y for x, y in zip(xs, ys) if lo < x <= hi]
        assert all(a >= b - 1e-12 for a, b in zip(seg, seg[1:]))


def test_what_if_data_shape_and_nonnegative_gain():
    w = dash.what_if_data(load_example_config(), today=0, day=3,
                          extra_minutes=60)
    assert w["day"] == 3
    assert w["extra_minutes"] == 60
    assert w["baseline_obj"] > 0
    assert w["delta"] >= -1e-9  # extra time cannot hurt
    assert isinstance(w["added"], list)
    assert isinstance(w["removed"], list)
    for entry in w["added"] + w["removed"]:
        d, exam, name, minutes = entry
        assert isinstance(d, int) and isinstance(minutes, int)


def test_fit_demo_data_fits_three_points():
    fd = dash.fit_demo_data()
    assert len(fd["pairs"]) == 3
    assert fd["S_fit"] == pytest.approx(1.946, rel=1e-3)
    # the fitted curve really is R(t) = exp(-t / S_fit)
    for x, r in zip(fd["xs"], fd["curve"]):
        assert r == pytest.approx(math.exp(-x / fd["S_fit"]))


def test_tool_scripts_compile():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for script in ("make_dashboard.py", "make_demo_gif.py"):
        path = os.path.join(root, "tools", script)
        assert os.path.exists(path)
        py_compile.compile(path, doraise=True)
