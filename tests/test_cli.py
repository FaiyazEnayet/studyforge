import csv
import json
import os

import pytest

from studyforge import cli
from studyforge import state as st

EXAMS = os.path.join(os.path.dirname(__file__), "..", "exams.yaml")


def run(args):
    cli.main(args)


def test_plan_creates_state(tmp_path):
    sp = str(tmp_path / "state.json")
    run(["--state", sp, "plan", "--exams", EXAMS, "--seed", "0"])
    assert os.path.exists(sp)
    with open(sp) as f:
        state = json.load(f)
    assert len(state["topics"]) == 15
    assert state["plan"], "plan should not be empty"
    cfg = state["config"]
    for day, sessions in state["plan"].items():
        assert int(day) not in cfg["rest_days"]
        assert sum(s["minutes"] for s in sessions) <= cfg["daily_budget_minutes"]


def test_log_updates_stability(tmp_path, capsys):
    sp = str(tmp_path / "state.json")
    run(["--state", sp, "plan", "--exams", EXAMS])
    before = st.load_state(sp)["topics"]["Rotation"]["S"]
    run(["--state", sp, "log", "--topic", "Rotation", "--minutes", "25",
         "--day", "0"])
    after = st.load_state(sp)["topics"]["Rotation"]
    assert after["last_review"] == 0
    assert after["S"] is not None
    assert len(st.load_state(sp)["history"]) == 1
    out = capsys.readouterr().out
    assert "Rotation" in out


def test_log_unknown_topic_fails(tmp_path):
    sp = str(tmp_path / "state.json")
    run(["--state", sp, "plan", "--exams", EXAMS])
    with pytest.raises(SystemExit):
        run(["--state", sp, "log", "--topic", "Nope", "--minutes", "25"])


def test_status_shows_readiness(tmp_path, capsys):
    sp = str(tmp_path / "state.json")
    run(["--state", sp, "plan", "--exams", EXAMS])
    run(["--state", sp, "status", "--today", "0"])
    out = capsys.readouterr().out
    assert "readiness" in out
    assert "Psychology Midterm" in out
    assert "Physics I Midterm" in out
    assert "Calculus III Final" in out


def test_replan_runs_and_keeps_history(tmp_path, capsys):
    sp = str(tmp_path / "state.json")
    run(["--state", sp, "plan", "--exams", EXAMS])
    run(["--state", sp, "log", "--topic", "Rotation", "--minutes", "25",
         "--day", "0"])
    run(["--state", sp, "replan", "--today", "2"])
    out = capsys.readouterr().out
    assert "Replanned from day 2" in out
    assert len(st.load_state(sp)["history"]) == 1


def test_what_if_runs(tmp_path, capsys):
    sp = str(tmp_path / "state.json")
    run(["--state", sp, "plan", "--exams", EXAMS])
    run(["--state", sp, "what-if", "--day", "1", "--extra-minutes", "60",
         "--today", "0"])
    out = capsys.readouterr().out
    assert "Baseline objective" in out
    assert "+60 min on day 1" in out


def test_export_csv_and_md(tmp_path):
    sp = str(tmp_path / "state.json")
    run(["--state", sp, "plan", "--exams", EXAMS])
    csvp = str(tmp_path / "plan.csv")
    mdp = str(tmp_path / "plan.md")
    run(["--state", sp, "export", "--format", "csv", "--out", csvp])
    run(["--state", sp, "export", "--format", "md", "--out", mdp])
    with open(csvp) as f:
        rows = list(csv.DictReader(f))
    assert rows and set(rows[0]) == {"day", "date", "exam", "topic", "minutes"}
    with open(mdp) as f:
        md = f.read()
    assert md.startswith("# Study plan")
    assert "## Day 0" in md
