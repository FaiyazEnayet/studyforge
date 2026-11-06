"""Exam config loading and JSON state handling."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
from datetime import date, datetime

import yaml

DEFAULT_STATE = "studyforge_state.json"

# Persisted-state schema versions. Version 1 is the current format:
# stable topic IDs with separate exam bindings, stamped on every save.
# A state with no "schema_version" key is legacy version 0 (pre-ST1):
# topics keyed by display name with the exam, deadline and weight
# stored on the topic itself. load_state migrates version 0 in memory.
SCHEMA_VERSION = 1


def _as_int(value, label: str) -> int:
    """A whole-number input, or a ValueError naming the fix."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"{label} must be a whole number, got {value!r}; "
            "use a plain number without quotes or a decimal point"
        )
    return value


def _as_positive_int(value, label: str) -> int:
    """A whole number greater than zero, or a ValueError naming the fix."""
    v = _as_int(value, label)
    if v <= 0:
        raise ValueError(
            f"{label} must be greater than 0, got {v}; "
            "use a positive number of minutes"
        )
    return v


def _as_finite_number(value, label: str) -> float:
    """A finite number, or a ValueError naming the fix."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"{label} must be a number, got {value!r}; "
            "use a plain number without quotes"
        )
    f = float(value)
    if not math.isfinite(f):
        raise ValueError(
            f"{label} must be finite, got {value!r}; "
            "replace it with an ordinary number"
        )
    return f


def validate_config(config: dict) -> dict:
    """Check a config dict (exams file shape) and return it normalized.

    Rejects an empty exam set, duplicate exam names, missing or
    negative exam days, exams with no topics, duplicate or colliding
    topic names, difficulty outside 1 to 5, nonfinite or non-positive
    weights, and zero or negative session lengths or daily caps. Every
    rejection names the bad value and what to fix.
    """
    if not isinstance(config, dict):
        raise ValueError(
            f"config must be a mapping of settings, got "
            f"{type(config).__name__}; check the exams file structure"
        )
    session_minutes = _as_positive_int(
        config.get("session_minutes", 25), "session_minutes")
    daily_cap = _as_positive_int(
        config.get("daily_budget_minutes", 120), "daily_budget_minutes")
    rest_days = config.get("rest_days", [])
    if not isinstance(rest_days, (list, tuple, set)):
        raise ValueError(
            f"rest_days must be a list of day offsets, got {rest_days!r}; "
            "write it as e.g. rest_days: [6, 13]"
        )
    rest = []
    for d in rest_days:
        dv = _as_int(d, "rest day")
        if dv < 0:
            raise ValueError(
                f"rest day must be 0 or later, got {dv}; "
                "rest days count forward from today (day 0)"
            )
        rest.append(dv)
    exams = config.get("exams")
    if not isinstance(exams, list) or not exams:
        raise ValueError(
            "config needs at least one exam: add an entry to the "
            "'exams' list with a name, a day and at least one topic"
        )
    seen_exams: set[str] = set()
    seen_ids: dict[str, str] = {}
    for ex in exams:
        if not isinstance(ex, dict):
            raise ValueError(
                f"each exam must be a mapping with a name, a day and a "
                f"topics list, got {ex!r}"
            )
        name = ex.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                "every exam needs a non-empty 'name'; add one to the "
                "exam entry"
            )
        if name in seen_exams:
            raise ValueError(
                f"duplicate exam {name!r}: exam names must be unique; "
                "rename one of the exams"
            )
        seen_exams.add(name)
        day = _as_int(ex.get("day"), f"day for exam {name!r}")
        if day < 0:
            raise ValueError(
                f"day for exam {name!r} must be 0 or later, got {day}; "
                "exam days count forward from today (day 0)"
            )
        topics = ex.get("topics")
        if not isinstance(topics, list) or not topics:
            raise ValueError(
                f"exam {name!r} needs at least one topic: add a topic "
                "with a name, a difficulty and a weight"
            )
        seen_in_exam: set[str] = set()
        for t in topics:
            if not isinstance(t, dict):
                raise ValueError(
                    f"each topic in exam {name!r} must be a mapping with "
                    f"a name, got {t!r}"
                )
            tname = t.get("name")
            if not isinstance(tname, str) or not tname.strip():
                raise ValueError(
                    f"every topic in exam {name!r} needs a non-empty "
                    "'name'; add one to the topic entry"
                )
            tid = topic_id(tname)
            if tid in seen_in_exam:
                raise ValueError(
                    f"duplicate topic {tname!r} in exam {name!r}: "
                    "list it once or rename one entry"
                )
            seen_in_exam.add(tid)
            if tid in seen_ids and seen_ids[tid] != tname:
                raise ValueError(
                    f"topic names {seen_ids[tid]!r} and {tname!r} both "
                    f"map to the same ID {tid!r}; rename one so their "
                    "IDs differ"
                )
            seen_ids[tid] = tname
            diff = _as_int(
                t.get("difficulty", 3),
                f"difficulty for topic {tname!r} in exam {name!r}")
            if not 1 <= diff <= 5:
                raise ValueError(
                    f"difficulty for topic {tname!r} in exam {name!r} "
                    f"must be between 1 and 5, got {diff}; use 1 "
                    "(easy) to 5 (hard)"
                )
            w = _as_finite_number(
                t.get("weight", 1.0),
                f"weight for topic {tname!r} in exam {name!r}")
            if w <= 0:
                raise ValueError(
                    f"weight for topic {tname!r} in exam {name!r} must "
                    f"be greater than 0, got {w}; give the topic a "
                    "positive weight"
                )
    return {
        "daily_budget_minutes": daily_cap,
        "session_minutes": session_minutes,
        "rest_days": sorted(set(rest)),
        "exams": exams,
    }


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
    """Parse an exams YAML file into a normalized config dict.

    Malformed input is rejected with an explicit error naming the
    exam or topic and what to fix, instead of a bare KeyError or a
    silently truncated number.
    """
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(
            f"exams file at {path} must contain settings and an 'exams' "
            "list" + ("; the file is empty" if cfg is None
                      else f", got {type(cfg).__name__}")
        )
    raw_exams = cfg.get("exams", [])
    if not isinstance(raw_exams, list):
        raise ValueError(
            f"'exams' in {path} must be a list of exams, got "
            f"{raw_exams!r}"
        )
    exams = []
    for ex in raw_exams:
        if not isinstance(ex, dict):
            raise ValueError(
                f"each exam in {path} must be a mapping with a name, a "
                f"day and a topics list, got {ex!r}"
            )
        name = ex.get("name")
        if name is None:
            raise ValueError(
                f"an exam in {path} is missing its 'name'; add a name "
                "to the exam entry"
            )
        if "day" not in ex:
            raise ValueError(
                f"exam {name!r} in {path} is missing its 'day'; add the "
                "exam day as a whole number of days from today"
            )
        raw_topics = ex.get("topics", [])
        if not isinstance(raw_topics, list):
            raise ValueError(
                f"'topics' for exam {name!r} in {path} must be a list, "
                f"got {raw_topics!r}"
            )
        topics = []
        for t in raw_topics:
            if not isinstance(t, dict) or "name" not in t:
                raise ValueError(
                    f"a topic in exam {name!r} in {path} is missing its "
                    "'name'; add a name to the topic entry"
                )
            topics.append(
                {
                    "name": str(t["name"]),
                    "difficulty": t.get("difficulty", 3),
                    "weight": t.get("weight", 1.0),
                }
            )
        exams.append(
            {"name": str(name), "day": ex["day"], "topics": topics}
        )
    return validate_config({
        "daily_budget_minutes": cfg.get("daily_budget_minutes", 120),
        "session_minutes": cfg.get("session_minutes", 25),
        "rest_days": cfg.get("rest_days", []),
        "exams": exams,
    })


def day0_today(state: dict) -> int:
    """Day offset of today relative to the plan's day zero."""
    try:
        d0 = datetime.strptime(state["day0"], "%Y-%m-%d").date()
    except (ValueError, TypeError, KeyError) as e:
        raise ValueError(
            f"state 'day0' must be a date like YYYY-MM-DD, got "
            f"{state.get('day0')!r}; fix the state file or rebuild it "
            "with `plan`"
        ) from e
    return (date.today() - d0).days


def new_state(config: dict) -> dict:
    """Fresh state from a config. Day zero is today.

    Knowledge topics get stable IDs (one memory state per topic),
    separately from exam-topic bindings (topic ID + weight per exam;
    the deadline comes from the exam). Two exams can share one topic
    without overwriting each other's objectives. A duplicate name
    within a single exam is rejected.
    """
    config = validate_config(config)
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
    try:
        return _migrate_v0(state)
    except (KeyError, TypeError, AttributeError) as e:
        raise ValueError(
            f"legacy state is malformed and cannot be migrated: {e}; "
            "fix the state file or rebuild it with `plan`"
        ) from e


def validate_state(state: dict) -> dict:
    """Check a persisted state dict (current schema) and return it.

    Rejects a malformed day0, an empty or duplicate exam set, exam
    days that are missing or negative, bindings that name a topic the
    state does not hold, topics bound to no exam, nonfinite or
    non-positive stability, difficulty outside 1 to 5, and history
    entries with a bad day, a recall outside 0 to 1, or a review day
    earlier than the same topic's previous review. History entries
    that name a topic by display name (migrated legacy states) are
    tolerated: only their shape and ranges are checked, matching the
    as-of rule that treats them as unusable for replay rather than
    as an error. Every rejection names the bad value and what to fix.
    """
    if not isinstance(state, dict):
        raise ValueError(
            f"state must be a JSON object, got {type(state).__name__}; "
            "fix the state file or rebuild it with `plan`"
        )
    try:
        datetime.strptime(state["day0"], "%Y-%m-%d")
    except (ValueError, TypeError, KeyError) as e:
        raise ValueError(
            f"state 'day0' must be a date like YYYY-MM-DD, got "
            f"{state.get('day0')!r}; fix the state file or rebuild it "
            "with `plan`"
        ) from e
    config = state.get("config")
    if not isinstance(config, dict):
        raise ValueError(
            "state is missing its 'config' object; fix the state file "
            "or rebuild it with `plan`"
        )
    _as_positive_int(
        config.get("session_minutes", 25), "session_minutes in state")
    _as_positive_int(
        config.get("daily_budget_minutes", 120),
        "daily_budget_minutes in state")
    rest_days = config.get("rest_days", [])
    if not isinstance(rest_days, (list, tuple, set)):
        raise ValueError(
            f"rest_days in state must be a list of day offsets, got "
            f"{rest_days!r}"
        )
    for d in rest_days:
        dv = _as_int(d, "rest day in state")
        if dv < 0:
            raise ValueError(
                f"rest day in state must be 0 or later, got {dv}"
            )
    topics = state.get("topics")
    if not isinstance(topics, dict):
        raise ValueError(
            "state is missing its 'topics' object; fix the state file "
            "or rebuild it with `plan`"
        )
    exams = config.get("exams")
    if not isinstance(exams, list) or not exams:
        raise ValueError(
            "state config needs at least one exam; fix the state file "
            "or rebuild it with `plan`"
        )
    seen_exams: set[str] = set()
    bound_ids: set[str] = set()
    for ex in exams:
        if not isinstance(ex, dict):
            raise ValueError(
                f"each exam in state config must be an object, got {ex!r}"
            )
        name = ex.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                "every exam in state config needs a non-empty 'name'"
            )
        if name in seen_exams:
            raise ValueError(
                f"duplicate exam {name!r} in state config: exam names "
                "must be unique; fix the state file or rebuild it with "
                "`plan`"
            )
        seen_exams.add(name)
        day = _as_int(ex.get("day"), f"day for exam {name!r} in state")
        if day < 0:
            raise ValueError(
                f"day for exam {name!r} in state must be 0 or later, "
                f"got {day}"
            )
        bindings = ex.get("topics")
        if not isinstance(bindings, list) or not bindings:
            raise ValueError(
                f"exam {name!r} in state needs at least one topic binding"
            )
        seen_in_exam: set[str] = set()
        for b in bindings:
            if not isinstance(b, dict):
                raise ValueError(
                    f"each binding in exam {name!r} must be an object, "
                    f"got {b!r}"
                )
            bid = b.get("id")
            if not isinstance(bid, str) or not bid:
                raise ValueError(
                    f"exam {name!r} in state has a binding without a "
                    "topic ID; fix the state file or rebuild it with "
                    "`plan`"
                )
            if bid in seen_in_exam:
                raise ValueError(
                    f"duplicate topic {bid!r} in exam {name!r} in state: "
                    "list it once"
                )
            seen_in_exam.add(bid)
            if bid not in topics:
                raise ValueError(
                    f"exam {name!r} in state binds unknown topic "
                    f"{bid!r}; add the topic to the state or remove "
                    "the binding"
                )
            w = _as_finite_number(
                b.get("weight", 1.0),
                f"weight for topic {bid!r} in exam {name!r} in state")
            if w <= 0:
                raise ValueError(
                    f"weight for topic {bid!r} in exam {name!r} in state "
                    f"must be greater than 0, got {w}"
                )
            bound_ids.add(bid)
    for tid, t in topics.items():
        if not isinstance(t, dict):
            raise ValueError(
                f"topic {tid!r} in state must be an object, got {t!r}"
            )
        if tid not in bound_ids:
            raise ValueError(
                f"topic {tid!r} in state is not bound to any exam; bind "
                "it to an exam or remove it from the state"
            )
        diff = _as_int(
            t.get("difficulty", 3), f"difficulty for topic {tid!r} in state")
        if not 1 <= diff <= 5:
            raise ValueError(
                f"difficulty for topic {tid!r} in state must be between "
                f"1 and 5, got {diff}"
            )
        s_val = t.get("S")
        if s_val is not None:
            s_num = _as_finite_number(
                s_val, f"stability for topic {tid!r} in state")
            if s_num <= 0:
                raise ValueError(
                    f"stability for topic {tid!r} in state must be "
                    f"greater than 0, got {s_num}"
                )
        lr = t.get("last_review")
        if lr is not None:
            lr_num = _as_int(
                lr, f"last_review for topic {tid!r} in state")
            if lr_num < 0:
                raise ValueError(
                    f"last_review for topic {tid!r} in state must be 0 "
                    f"or later, got {lr_num}"
                )
    history = state.get("history", [])
    if not isinstance(history, list):
        raise ValueError(
            f"history in state must be a list, got {type(history).__name__}"
        )
    last_day_by_topic: dict[str, int] = {}
    for i, h in enumerate(history):
        if not isinstance(h, dict):
            raise ValueError(
                f"history entry {i} in state must be an object, got {h!r}"
            )
        day = _as_int(h.get("day"), f"day for history entry {i} in state")
        if day < 0:
            raise ValueError(
                f"history entry {i} in state has day {day}; review days "
                "must be 0 or later"
            )
        minutes = _as_int(
            h.get("minutes", 0), f"minutes for history entry {i} in state")
        if minutes < 0:
            raise ValueError(
                f"history entry {i} in state has minutes {minutes}; "
                "logged minutes must be 0 or later"
            )
        recall = h.get("recall")
        if recall is not None:
            r_num = _as_finite_number(
                recall, f"recall for history entry {i} in state")
            if not 0.0 <= r_num <= 1.0:
                raise ValueError(
                    f"recall for history entry {i} in state must be "
                    f"between 0 and 1, got {r_num}; log the fraction "
                    "actually remembered"
                )
        interval = h.get("interval")
        if interval is not None:
            iv = _as_int(
                interval, f"interval for history entry {i} in state")
            if iv < 0:
                raise ValueError(
                    f"history entry {i} in state has interval {iv}; "
                    "intervals must be 0 or later"
                )
        topic = h.get("topic")
        if topic is not None:
            prev = last_day_by_topic.get(topic)
            if prev is not None and day < prev:
                raise ValueError(
                    f"history for topic {topic!r}: review on day {day} "
                    f"comes before the previous review on day {prev}; "
                    "history entries must be in day order"
                )
            last_day_by_topic[topic] = day
    plan = state.get("plan", {})
    if not isinstance(plan, dict):
        raise ValueError(
            f"plan in state must be an object, got {type(plan).__name__}"
        )
    return state


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
    """Write `state` to `path`, preserving a backup of the previous file.

    The previous state file, when one exists, is copied to
    ``<path>.bak`` before the new state is written, so a save never
    destroys the last good state: the backup always holds the
    immediately previous saved state and nothing older. A first save
    creates no backup. The backup is a byte copy of the previous
    file, so it round-trips through load_state exactly as the file
    it came from did.
    """
    state["schema_version"] = SCHEMA_VERSION
    if os.path.exists(path):
        shutil.copyfile(path, path + ".bak")
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
    return validate_state(migrate_state(state))


def _asof_topic_state(state: dict, tid: str, day: int):
    """One topic's memory state as of `day`, or None if unavailable.

    Mirrors the planner's as-of rule: only reviews strictly before
    `day` may inform the estimate. If every logged review precedes
    `day`, the stored state already is the as-of state. Otherwise the
    state is rebuilt by replaying the topic's ID-keyed history entries
    before `day` through the memory model's update rule, using the
    observed recall where one was logged.

    Returns None when the recorded history does not cover the stored
    state: a migrated legacy state, whose history entries name topics
    by display name, has no ID-keyed events backing its (S,
    last_review), so the as-of state cannot be reconstructed and the
    caller must mark the estimate unavailable rather than evaluating
    it from today's state.
    """
    from .memory import TopicState

    t = state["topics"][tid]
    lr = t["last_review"]
    if lr is None or lr < day:
        return TopicState(
            name=t["name"],
            difficulty=t["difficulty"],
            S=t["S"],
            last_review=lr,
        )
    events = [
        (int(h["day"]), h.get("recall"))
        for h in state.get("history", [])
        if h.get("topic") == tid
    ]
    if not events or max(d for d, _ in events) < lr:
        return None
    ts = TopicState(name=t["name"], difficulty=t["difficulty"])
    for d, r in sorted(events):
        if d >= day:
            break
        ts.review(d, R=r)
    return ts


def readiness(state: dict, today: int) -> dict[str, dict]:
    """Per-exam readiness: weight normalized recall across its topics.

    For each exam, readiness = sum(weight * predicted recall at exam day)
    / sum(weights), using no further study. A topic bound to several
    exams contributes to each with that binding's weight and deadline,
    over the one shared memory state.

    The estimate is as-of the exam day. For an exam still in the
    future, the current memory state is the as-of state. For an exam
    on or before `today`, the estimate is historical: each topic's
    state is rebuilt as of the exam day (see _asof_topic_state), so
    reviews logged after the exam never leak into it. When a bound
    topic's as-of state cannot be reconstructed, the exam's readiness
    is None (unavailable) instead of a number evaluated from today's
    state. Returns {exam_name: {"readiness": 0..1 or None,
    "days_left": int, "topics": n}}.
    """
    from .memory import TopicState

    out: dict[str, dict] = {}
    for ex in state["config"]["exams"]:
        num, den = 0.0, 0.0
        unavailable = False
        for b in ex["topics"]:
            if ex["day"] > today:
                t = state["topics"][b["id"]]
                ts = TopicState(
                    name=t["name"],
                    difficulty=t["difficulty"],
                    S=t["S"],
                    last_review=t["last_review"],
                )
            else:
                ts = _asof_topic_state(state, b["id"], ex["day"])
                if ts is None:
                    unavailable = True
                    break
            num += b["weight"] * ts.recall_on(ex["day"])
            den += b["weight"]
        out[ex["name"]] = {
            "readiness": None if unavailable else (num / den if den else 0.0),
            "days_left": ex["day"] - today,
            "topics": len(ex["topics"]),
        }
    return out
