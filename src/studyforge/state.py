"""Exam config loading and JSON state handling."""

from __future__ import annotations

import json
import os
from datetime import date, datetime

import yaml

DEFAULT_STATE = "studyforge_state.json"


def load_config(path: str) -> dict:
    """Parse an exams YAML file into a normalized config dict."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    exams = []
    for ex in cfg.get("exams", []):
        topics = []
        for t in ex.get("topics", []):
            topics.append(
                {
                    "name": str(t["name"]),
                    "difficulty": int(t.get("difficulty", 3)),
                    "weight": float(t.get("weight", 1.0)),
                }
            )
        exams.append(
            {"name": str(ex["name"]), "day": int(ex["day"]), "topics": topics}
        )
    return {
        "daily_budget_minutes": int(cfg.get("daily_budget_minutes", 120)),
        "session_minutes": int(cfg.get("session_minutes", 25)),
        "rest_days": sorted(set(int(d) for d in cfg.get("rest_days", []))),
        "exams": exams,
    }


def day0_today(state: dict) -> int:
    """Day offset of today relative to the plan's day zero."""
    d0 = datetime.strptime(state["day0"], "%Y-%m-%d").date()
    return (date.today() - d0).days


def new_state(config: dict) -> dict:
    """Fresh state from a config. Day zero is today."""
    topics: dict[str, dict] = {}
    for ex in config["exams"]:
        for t in ex["topics"]:
            topics[t["name"]] = {
                "exam": ex["name"],
                "exam_day": ex["day"],
                "difficulty": t["difficulty"],
                "weight": t["weight"],
                "S": None,  # filled from difficulty on first use
                "last_review": None,
            }
    return {
        "day0": date.today().isoformat(),
        "config": config,
        "topics": topics,
        "history": [],
        "plan": {},
    }


def save_state(state: dict, path: str = DEFAULT_STATE) -> None:
    with open(path, "w") as f:
        json.dump(state, f, indent=1)


def load_state(path: str = DEFAULT_STATE) -> dict:
    if not os.path.exists(path):
        raise SystemExit(f"no state file at {path}; run `plan` first")
    with open(path) as f:
        return json.load(f)


def readiness(state: dict, today: int) -> dict[str, dict]:
    """Per-exam readiness: weight normalized recall across its topics.

    For each exam, readiness = sum(weight * predicted recall at exam day)
    / sum(weights), using current memory state and no further study.
    Returns {exam_name: {"readiness": 0..1, "days_left": int, "topics": n}}.
    """
    from .memory import TopicState

    out: dict[str, dict] = {}
    for ex in state["config"]["exams"]:
        num, den = 0.0, 0.0
        for tname, t in state["topics"].items():
            if t["exam"] != ex["name"]:
                continue
            ts = TopicState(
                name=tname,
                difficulty=t["difficulty"],
                S=t["S"],
                last_review=t["last_review"],
            )
            num += t["weight"] * ts.recall_on(ex["day"])
            den += t["weight"]
        out[ex["name"]] = {
            "readiness": (num / den) if den else 0.0,
            "days_left": ex["day"] - today,
            "topics": sum(1 for t in state["topics"].values() if t["exam"] == ex["name"]),
        }
    return out
