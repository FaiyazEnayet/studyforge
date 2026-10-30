"""Exam config loading and JSON state handling."""

from __future__ import annotations

import json
import os
import re
from datetime import date, datetime

import yaml

DEFAULT_STATE = "studyforge_state.json"

# Persisted-state schema versions. Version 1 is the current format:
# stable topic IDs with separate exam bindings, stamped on every save.
# A state with no "schema_version" key is legacy version 0 (pre-ST1):
# topics keyed by display name with the exam, deadline and weight
# stored on the topic itself. load_state migrates version 0 in memory.
SCHEMA_VERSION = 1


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
        "schema_version": SCHEMA_VERSION,
        "day0": date.today().isoformat(),
        "config": new_config,
        "topics": topics,
        "history": [],
        "plan": {},
    }


def migrate_state(state: dict) -> dict:
    """Return `state` in the current schema version, migrating in memory.

    Legacy version 0 (no "schema_version" key) predates stable topic
    IDs: topics were keyed by display name with the exam, deadline and
    weight stored on the topic, and a topic shared by two exams kept
    only the last exam's association. Migration rebuilds the bindings
    from the config's exam topic lists (the same source new_state
    uses), so a shared topic becomes one memory state with one binding
    per exam and no binding is lost. Migrating a current state changes
    nothing, and migrating twice is the same as migrating once. The
    input dict is never mutated and nothing is written to disk here:
    a state that cannot be migrated raises ValueError, and any file it
    came from is left exactly as it was.
    """
    if not isinstance(state, dict):
        raise ValueError(
            f"state must be a JSON object, got {type(state).__name__}"
        )
    version = state.get("schema_version", 0)
    if version == SCHEMA_VERSION:
        return state
    if version != 0:
        raise ValueError(
            f"unsupported state schema_version {version!r}; this build "
            f"understands 0 (legacy) through {SCHEMA_VERSION}"
        )
    return _migrate_v0(state)


def _migrate_v0(state: dict) -> dict:
    """Migrate a legacy (version 0) state dict to the current format."""
    config = state.get("config")
    legacy_topics = state.get("topics")
    if not isinstance(config, dict) or not isinstance(legacy_topics, dict):
        raise ValueError(
            "legacy state is missing its 'config' or 'topics' object; "
            "cannot migrate"
        )
    if "day0" not in state:
        raise ValueError("legacy state is missing 'day0'; cannot migrate")

    def memory_for(name: str, difficulty) -> dict:
        mem = legacy_topics.get(name)
        if mem is None:
            return {
                "name": name,
                "difficulty": difficulty,
                "S": None,
                "S_source": "default",
                "last_review": None,
            }
        if not isinstance(mem, dict):
            raise ValueError(
                f"legacy topic {name!r} is not an object; cannot migrate"
            )
        return {
            "name": name,
            "difficulty": mem.get("difficulty", difficulty),
            "S": mem.get("S"),
            "S_source": mem.get("S_source", "default"),
            "last_review": mem.get("last_review"),
        }

    topics: dict[str, dict] = {}
    exams = []
    for ex in config.get("exams", []):
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
                topics[tid] = memory_for(t["name"], t.get("difficulty", 3))
            bindings.append({"id": tid, "weight": t.get("weight", 1.0)})
        exams.append({"name": ex["name"], "day": ex["day"], "topics": bindings})
    # A legacy topic no exam references still carries memory state;
    # keep it rather than dropping logged progress.
    for name, mem in legacy_topics.items():
        tid = topic_id(name)
        if tid not in topics:
            fallback = mem.get("difficulty") if isinstance(mem, dict) else None
            topics[tid] = memory_for(name, fallback)
    new_config = dict(config)
    new_config["exams"] = exams
    return {
        "schema_version": SCHEMA_VERSION,
        "day0": state["day0"],
        "config": new_config,
        "topics": topics,
        "history": state.get("history", []),
        "plan": state.get("plan", {}),
    }


def save_state(state: dict, path: str = DEFAULT_STATE) -> None:
    state["schema_version"] = SCHEMA_VERSION
    with open(path, "w") as f:
        json.dump(state, f, indent=1)


def load_state(path: str = DEFAULT_STATE) -> dict:
    if not os.path.exists(path):
        raise SystemExit(f"no state file at {path}; run `plan` first")
    with open(path) as f:
        try:
            state = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"state file at {path} is not valid JSON: {e}"
            ) from e
    return migrate_state(state)


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
