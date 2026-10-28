"""Stage 3 studyforge A02: generator/data contract and evidence provenance.

Failing-first regression tests for the defects on main (HEAD 03c1bce):
- dashboard.plan_data topic_info has no exam_day, so
  tools/make_dashboard.py fig_forgetting_curves crashes with KeyError.
- tools/make_demo_gif.py looks up topic_info["Rotation"], but topic
  keys are normalized IDs ("rotation"), so it crashes with KeyError.
- dashboard.fit_demo_data returns hardcoded example pairs with no
  provenance, while tools/make_dashboard.py fig_fit labels them
  "your logged reviews", a personal-history claim with no records.
"""

import os

import pytest
import yaml

from studyforge import dashboard as dash
from studyforge.planner import PlanTopic, simulate
from studyforge.state import topic_id


def load_example_config():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "exams.yaml")) as f:
        cfg = yaml.safe_load(f)
    cfg["rest_days"] = sorted(set(cfg.get("rest_days", [])))
    return cfg


def test_topic_info_supplies_binding_aware_exam_days():
    data = dash.plan_data(load_example_config())
    for name, info in data["topic_info"].items():
        binding_days = sorted(d for _, _, d in info["bindings"])
        assert info["exam_days"] == binding_days
        # The single exam_day a curve view needs is the latest binding
        # deadline; every session is scheduled before it.
        assert info["exam_day"] == binding_days[-1]
        assert all(d < info["exam_day"] for d in info["days"])


def test_topic_exam_day_rejects_schema_mismatch():
    data = dash.plan_data(load_example_config())
    info = dict(data["topic_info"]["rotation"])
    assert dash.topic_exam_day(info) == 14
    missing = {k: v for k, v in info.items() if k != "exam_day"}
    with pytest.raises(ValueError, match="exam_day"):
        dash.topic_exam_day(missing)
    inconsistent = dict(info, exam_day=999)
    with pytest.raises(ValueError, match="exam_day"):
        dash.topic_exam_day(inconsistent)


def test_generator_topic_lookup_uses_normalized_id():
    data = dash.plan_data(load_example_config())
    # The GIF topic is "Rotation" by display name; topic_info is keyed
    # by the normalized ID, so the lookup must go through topic_id.
    assert topic_id("Rotation") == "rotation"
    info = data["topic_info"][topic_id("Rotation")]
    assert info["exam_day"] == 14


def test_forgetting_curve_data_agrees_with_engine_fixture():
    data = dash.plan_data(load_example_config())
    info = data["topic_info"]["rotation"]
    traj = dash.recall_trajectory("rotation", info["days"], info["S0"],
                                  info["exam_day"], data["horizon"])
    _, recalls = simulate(
        PlanTopic("rotation", info["bindings"], info["S0"], None),
        info["days"])
    assert traj["recall_at_exam"] == pytest.approx(recalls[0])
    assert traj["S_final"] == pytest.approx(info["S_final"])


def test_fit_demo_data_is_labeled_illustrative_not_logged():
    fd = dash.fit_demo_data()
    assert fd["S_fit"] == pytest.approx(1.946, rel=1e-3)
    prov = fd["provenance"]
    assert prov["kind"] == "illustrative_fixture"
    assert prov["logged_records"] == 0
    assert "logged" not in prov["kind"]
    assert prov["source"] != "logged history"


def test_generators_execute_end_to_end(tmp_path):
    """Smoke test: both generators run and write real output.

    Skipped where the dashboard extras are not installed (CI installs
    only the dev extra); when present, this executes both generators
    instead of only compiling them. Output goes to tmp_path so the
    repo docs/ outputs are never rewritten by a test run.
    """
    pytest.importorskip("plotly")
    pytest.importorskip("matplotlib")
    import sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(root, "tools"))
    import make_dashboard
    import make_demo_gif

    make_dashboard.OUT = str(tmp_path / "dashboard.html")
    make_dashboard.main()
    assert os.path.getsize(make_dashboard.OUT) > 1000

    make_demo_gif.OUT = str(tmp_path / "demo.gif")
    make_demo_gif.main()
    assert os.path.getsize(make_demo_gif.OUT) > 1000
