"""Exam config loading and JSON state handling."""

from __future__ import annotations

import json
import os
import re
from datetime import date, datetime

import yaml

DEFAULT_STATE = "studyforge_state.json"


def topic_id(name: str) -> str:
    """Stable ID for a knowledge topic, derived from its display name.

    The same name always maps to the same ID, so a topic shared by two
    exams is one knowledge state with two exam bindings. Names that
    slug to nothing are rejected.
    """
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    if not slug:
        raise ValueError(f"topic name {name!r} has no usable ID")
    return slug


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
    """Fresh state from a config. Day zero is today.

    Knowledge topics get stable IDs (one memory state per topic),
    separately from exam-topic bindings (topic ID + weight per exam;
    the deadline comes from the exam). Two exams can share one topic
    without overwriting each other's objectives. A duplicate name
    within a single exam is rejected.
    """
    topics: dict[str, dict] = {}
    exams = []
    for ex in config["exams"]:
        bindings = []
        seen: set[str] = set()
        for t in ex["topics"]:
            tid = topic_id(t["name"])
            if tid in seen:
                raise ValueError(
                    f"duplicate topic {t['name']!r} in exam {ex['name']!r}: "
                    "list it once or rename one entry"
                )
            seen.add(tid)
            if tid not in topics:
                topics[tid] = {
                    "name": t["name"],
                    "difficulty": t["difficulty"],
                    "S": None,  # filled from difficulty on first use
                    "S_source": "default",  # or "fitted" once logs personalize it
                    "last_review": None,
                }
            bindings.append({"id": tid, "weight": t["weight"]})
        exams.append({"name": ex["name"], "day": ex["day"], "topics": bindings})
    new_config = dict(config)
    new_config["exams"] = exams
    return {
        "day0": date.today().isoformat(),
        "config": new_config,
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
    A topic bound to several exams contributes to each with that
    binding's weight and deadline, over the one shared memory state.
    Returns {exam_name: {"readiness": 0..1, "days_left": int, "topics": n}}.
    """
    from .memory import TopicState

    out: dict[str, dict] = {}
    for ex in state["config"]["exams"]:
        num, den = 0.0, 0.0
        for b in ex["topics"]:
            t = state["topics"][b["id"]]
            ts = TopicState(
                name=t["name"],
                difficulty=t["difficulty"],
                S=t["S"],
                last_review=t["last_review"],
            )
            num += b["weight"] * ts.recall_on(ex["day"])
            den += b["weight"]
        out[ex["name"]] = {
            "readiness": (num / den) if den else 0.0,
            "days_left": ex["day"] - today,
            "topics": len(ex["topics"]),
        }
    return out
