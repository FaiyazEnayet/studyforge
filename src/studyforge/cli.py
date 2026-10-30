"""studyforge command line interface."""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timedelta

from . import state as st
from .memory import TopicState, fit_stability, initial_stability
from .planner import PlanTopic, build_plan, to_schedule


def _plan_topics(state: dict) -> list[PlanTopic]:
    """One PlanTopic per knowledge topic, with all its exam bindings.

    A topic shared by two exams keeps one memory state but contributes
    to the objective at each bound exam's deadline. Each PlanTopic
    also carries the topic's logged review history (ID-keyed entries
    from the state history) and difficulty, so the planner can rebuild
    the memory state as of an earlier exam day by event replay.
    History entries that name a topic by display name (migrated
    legacy states) do not match any topic ID: those topics get an
    empty recorded history, and an as-of estimate that would need
    their events is marked unavailable instead of fabricated.
    """
    by_id = {ex["name"]: ex for ex in state["config"]["exams"]}
    history: dict[str, list[tuple[int, float | None]]] = {}
    for h in state.get("history", []):
        history.setdefault(h.get("topic"), []).append(
            (int(h["day"]), h.get("recall")))
    pts = []
    for tid, t in state["topics"].items():
        bindings = []
        for ex in state["config"]["exams"]:
            for b in ex["topics"]:
                if b["id"] == tid:
                    bindings.append((ex["name"], b["weight"], ex["day"]))
        S = t["S"] if t["S"] is not None else initial_stability(t["difficulty"])
        pts.append(PlanTopic(tid, bindings, S, t["last_review"],
                             difficulty=t["difficulty"],
                             history=history.get(tid, [])))
    return pts


def _resolve_topic(state: dict, ref: str) -> str:
    """Resolve a CLI topic reference (ID or display name) to a topic ID."""
    if ref in state["topics"]:
        return ref
    matches = [tid for tid, t in state["topics"].items()
               if t["name"].lower() == ref.lower()]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise SystemExit(f"unknown topic: {ref}")
    raise SystemExit(f"ambiguous topic {ref!r}: matches {matches}")


def _date_of(state: dict, day: int) -> str:
    d0 = datetime.strptime(state["day0"], "%Y-%m-%d").date()
    return (d0 + timedelta(days=day)).isoformat()


def _exams_of(state: dict, topic_id: str) -> list[str]:
    """Names of exams a topic is bound to (a shared topic lists several)."""
    return [ex["name"] for ex in state["config"]["exams"]
            if any(b["id"] == topic_id for b in ex["topics"])]


def _store_plan(state: dict, sched: dict[int, list[tuple[str, int]]]) -> None:
    state["plan"] = {
        str(d): [{"topic": n, "minutes": m} for n, m in sessions]
        for d, sessions in sorted(sched.items())
    }


def _print_schedule(state: dict, sched: dict[int, list[tuple[str, int]]],
                    horizon: int, rest_days: set[int]) -> None:
    for d in range(horizon):
        sessions = sched.get(d, [])
        if d in rest_days:
            print(f"Day {d} ({_date_of(state, d)}): rest")
        elif not sessions:
            print(f"Day {d} ({_date_of(state, d)}): no sessions")
        else:
            print(f"Day {d} ({_date_of(state, d)}):")
            for name, minutes in sessions:
                print(f"  {",".join(_exams_of(state, name))} / {name}: {minutes} min")


def _objective_of(state: dict, plan_days: dict[str, list[int]]) -> float:
    from .planner import objective

    return objective(_plan_topics(state), plan_days)


def cmd_plan(args: argparse.Namespace) -> None:
    cfg = st.load_config(args.exams)
    state = st.new_state(cfg)
    topics = _plan_topics(state)
    horizon = max(t.latest_exam_day for t in topics)
    rest = set(cfg["rest_days"])
    plan, obj = build_plan(
        topics, horizon, cfg["daily_budget_minutes"],
        cfg["session_minutes"], rest, start_day=0, seed=args.seed,
    )
    sched = to_schedule(plan, cfg["session_minutes"])
    _store_plan(state, sched)
    st.save_state(state, args.state)
    n_sessions = sum(len(v) for v in plan.values())
    print(f"Study plan: {n_sessions} sessions, objective {obj:.3f}")
    print(f"Budget: {cfg['daily_budget_minutes']} min/day, "
          f"sessions of {cfg['session_minutes']} min")
    _print_schedule(state, sched, horizon, rest)


def cmd_log(args: argparse.Namespace) -> None:
    state = st.load_state(args.state)
    tid = _resolve_topic(state, args.topic)
    if args.recall is not None and not 0.0 <= args.recall <= 1.0:
        raise SystemExit("--recall must be between 0 and 1")
    today = args.day if args.day is not None else st.day0_today(state)
    t = state["topics"][tid]
    prev_last = t["last_review"]
    ts = TopicState(
        name=tid,
        difficulty=t["difficulty"],
        S=t["S"],
        last_review=t["last_review"],
    )
    old_S = ts.S
    R = ts.review(today, R=args.recall)
    t["S"] = ts.S
    t["last_review"] = ts.last_review
    interval = None if prev_last is None else today - prev_last
    state["history"].append(
        {"day": today, "topic": tid, "minutes": args.minutes,
         "interval": interval, "recall": args.recall}
    )
    # Refit this topic's personal stability from its logged history.
    pairs = [(h.get("interval"), h.get("recall")) for h in state["history"]
             if h["topic"] == tid]
    n_pairs = sum(1 for iv, rc in pairs
                  if iv is not None and rc is not None and iv > 0)
    S_fit = fit_stability(pairs)
    if S_fit is not None:
        t["S"] = S_fit
        t["S_source"] = "fitted"
    st.save_state(state, args.state)
    print(f"Logged {args.minutes} min on {t['name']} (day {today}).")
    print(f"Retrievability at review: {R:.0%}. "
          f"Stability: {old_S:.2f} -> {ts.S:.2f} days.")
    if S_fit is not None:
        print(f"Fitted your personal stability from {n_pairs} recall "
              f"data points: {S_fit:.2f} days.")


def _rebuild(state: dict, today: int, seed: int,
             extra: dict[int, int] | None = None):
    """Rebuild the plan from today. extra maps day -> added minutes of cap."""
    cfg = state["config"]
    topics = _plan_topics(state)
    horizon = max(t.latest_exam_day for t in topics)
    rest = set(cfg["rest_days"])
    if not extra:
        plan, obj = build_plan(
            topics, horizon, cfg["daily_budget_minutes"],
            cfg["session_minutes"], rest, start_day=today, seed=seed,
        )
        return plan, obj, horizon
    # per-day caps for the what-if run
    from .planner import greedy, local_search, objective

    caps = {d: cfg["daily_budget_minutes"] + extra.get(d, 0)
            for d in range(today, horizon)}

    def greedy_caps(topics, horizon, session_min, rest_days, start_day):
        plan = {tp.name: [] for tp in topics}
        used: dict[int, int] = {}
        ordered = sorted(topics, key=lambda t: t.name)
        from .planner import MIN_GAIN, MAX_PER_TOPIC_PER_DAY, _day_has_room
        while True:
            base = objective(topics, plan, as_of=today)
            best = None
            for tp in ordered:
                for d in range(start_day, min(horizon, tp.latest_exam_day)):
                    cap = caps.get(d, cfg["daily_budget_minutes"])
                    if not _day_has_room(used, d, session_min, cap, rest_days):
                        continue
                    if plan[tp.name].count(d) >= MAX_PER_TOPIC_PER_DAY:
                        continue
                    plan[tp.name].append(d)
                    gain = objective(topics, plan, as_of=today) - base
                    plan[tp.name].pop()
                    if gain > MIN_GAIN and (best is None or gain > best[0] + MIN_GAIN):
                        best = (gain, tp.name, d)
            if best is None:
                break
            _, name, d = best
            plan[name].append(d)
            plan[name].sort()
            used[d] = used.get(d, 0) + session_min
        return plan

    plan = greedy_caps(topics, horizon, cfg["session_minutes"], rest, today)
    # Same local search as the standard path, but with the per-day caps,
    # so the what-if comparison stays apples to apples.
    plan = local_search(topics, plan, horizon, cfg["daily_budget_minutes"],
                        cfg["session_minutes"], rest, today, caps=caps,
                        as_of=today)
    return plan, objective(topics, plan, as_of=today), horizon


def _diff_plans(old_sched: dict[int, list[tuple[str, int]]],
                new_sched: dict[int, list[tuple[str, int]]],
                today: int) -> tuple[list, list]:
    def multiset(sched):
        items = []
        for d, sessions in sched.items():
            if d < today:
                continue
            for name, minutes in sessions:
                items.append((d, name, minutes))
        return sorted(items)

    old, new = multiset(old_sched), multiset(new_sched)
    old_c, new_c = old[:], new[:]
    for item in old:
        if item in new_c:
            new_c.remove(item)
    for item in new:
        if item in old_c:
            old_c.remove(item)
    return new_c, old_c  # added, removed


def cmd_replan(args: argparse.Namespace) -> None:
    state = st.load_state(args.state)
    today = args.today if args.today is not None else st.day0_today(state)
    cfg = state["config"]
    old_sched = {
        int(d): [(s["topic"], s["minutes"]) for s in sessions]
        for d, sessions in state.get("plan", {}).items()
    }
    plan, obj, horizon = _rebuild(state, today, args.seed)
    sched = to_schedule(plan, cfg["session_minutes"])
    _store_plan(state, sched)
    st.save_state(state, args.state)
    added, removed = _diff_plans(old_sched, sched, today)
    n_sessions = sum(len(v) for v in plan.values())
    print(f"Replanned from day {today}: {n_sessions} sessions, "
          f"objective {obj:.3f}")
    if added:
        print("Added sessions:")
        for d, name, minutes in added:
            print(f"  Day {d}: {",".join(_exams_of(state, name))} / {name}: {minutes} min")
    if removed:
        print("Removed sessions:")
        for d, name, minutes in removed:
            print(f"  Day {d}: {",".join(_exams_of(state, name))} / {name}: {minutes} min")
    if not added and not removed:
        print("No changes to the remaining plan.")


def cmd_status(args: argparse.Namespace) -> None:
    state = st.load_state(args.state)
    today = args.today if args.today is not None else st.day0_today(state)
    print(f"Status on day {today} ({_date_of(state, today)}):")
    print(f"{'Topic':32s} {'Exams':22s} {'S(days)':>7s} "
          f"{'Recall':>7s} {'Exam in':>7s} {'R@exam':>7s} {'S from':>8s}")
    # earliest bound deadline per topic for the at-a-glance columns
    deadline = {}
    for ex in state["config"]["exams"]:
        for b in ex["topics"]:
            if b["id"] not in deadline or ex["day"] < deadline[b["id"]]:
                deadline[b["id"]] = ex["day"]
    for tid in sorted(state["topics"], key=lambda i: state["topics"][i]["name"]):
        t = state["topics"][tid]
        ts = TopicState(name=tid, difficulty=t["difficulty"], S=t["S"],
                        last_review=t["last_review"])
        r_now = ts.recall_on(today)
        d_day = deadline.get(tid, today)
        r_exam = ts.recall_on(d_day)
        S = t["S"] if t["S"] is not None else initial_stability(t["difficulty"])
        src = t.get("S_source", "default")
        exams = ",".join(_exams_of(state, tid))[:22]
        print(f"{t['name']:32s} {exams:22s} {S:7.2f} "
              f"{r_now:7.0%} {d_day - today:7d} {r_exam:7.0%} "
              f"{src:>8s}")
    print()
    n_fit = sum(1 for t in state["topics"].values()
                if t.get("S_source") == "fitted")
    n_tot = len(state["topics"])
    print(f"Stability: {n_fit}/{n_tot} topics fitted from your logs. "
          f"The rest use defaults. Log sessions with --recall to "
          f"personalize more topics.")
    print()
    print("Per-exam readiness (weighted recall at exam time, no more study):")
    for exam, info in st.readiness(state, today).items():
        r = info["readiness"]
        r_txt = "n/a (as-of history unavailable)" if r is None else f"{r:.0%}"
        print(f"  {exam}: {r_txt} "
              f"({info['topics']} topics, {info['days_left']} days left)")


def cmd_what_if(args: argparse.Namespace) -> None:
    state = st.load_state(args.state)
    today = args.today if args.today is not None else st.day0_today(state)
    cfg = state["config"]
    topics = _plan_topics(state)
    horizon = max(t.latest_exam_day for t in topics)
    rest = set(cfg["rest_days"])
    base_plan, base_obj = build_plan(
        topics, horizon, cfg["daily_budget_minutes"],
        cfg["session_minutes"], rest, start_day=today, seed=args.seed,
    )
    extra = {args.day: args.extra_minutes}
    new_plan, new_obj, _ = _rebuild(state, today, args.seed, extra=extra)
    base_sched = to_schedule(base_plan, cfg["session_minutes"])
    new_sched = to_schedule(new_plan, cfg["session_minutes"])
    added, removed = _diff_plans(base_sched, new_sched, today)
    print(f"Baseline objective: {base_obj:.3f}")
    print(f"With +{args.extra_minutes} min on day {args.day}: {new_obj:.3f} "
          f"({new_obj - base_obj:+.3f})")
    if added:
        print("Sessions added by the extra time:")
        for d, name, minutes in added:
            print(f"  Day {d}: {",".join(_exams_of(state, name))} / {name}: {minutes} min")
    if removed:
        print("Sessions displaced:")
        for d, name, minutes in removed:
            print(f"  Day {d}: {",".join(_exams_of(state, name))} / {name}: {minutes} min")
    if not added and not removed:
        print("No change: the extra time buys nothing at the margin.")


def cmd_export(args: argparse.Namespace) -> None:
    state = st.load_state(args.state)
    cfg = state["config"]
    sched = {
        int(d): [(s["topic"], s["minutes"]) for s in sessions]
        for d, sessions in state.get("plan", {}).items()
    }
    rows = []
    for d in sorted(sched):
        for name, minutes in sched[d]:
            rows.append({
                "day": d,
                "date": _date_of(state, d),
                "exam": ",".join(_exams_of(state, name)),
                "topic": name,
                "minutes": minutes,
            })
    if args.format == "csv":
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["day", "date", "exam", "topic", "minutes"])
            w.writeheader()
            w.writerows(rows)
    else:
        with open(args.out, "w") as f:
            f.write("# Study plan\n\n")
            for d in sorted(sched):
                f.write(f"## Day {d} ({_date_of(state, d)})\n\n")
                for name, minutes in sched[d]:
                    exams = ",".join(_exams_of(state, name))
                    f.write(f"- {exams} / {name}: {minutes} min\n")
                f.write("\n")
    print(f"Exported {len(rows)} sessions to {args.out}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="studyforge",
                                description="Exam study scheduling under a "
                                            "forgetting curve.")
    p.add_argument("--state", default=st.DEFAULT_STATE,
                   help="path to the JSON state file")
    sub = p.add_subparsers(dest="command", required=True)

    pp = sub.add_parser("plan", help="build a study plan from an exams file")
    pp.add_argument("--exams", required=True, help="path to exams YAML")
    pp.add_argument("--seed", type=int, default=0)
    pp.set_defaults(func=cmd_plan)

    pl = sub.add_parser("log", help="log a completed study session")
    pl.add_argument("--topic", required=True)
    pl.add_argument("--minutes", type=int, required=True)
    pl.add_argument("--day", type=int, default=None,
                    help="day offset (default: today)")
    pl.add_argument("--recall", type=float, default=None,
                    help="fraction remembered at review, 0 to 1 "
                         "(personalizes this topic's stability)")
    pl.set_defaults(func=cmd_log)

    pr = sub.add_parser("replan", help="rebuild the plan from today")
    pr.add_argument("--today", type=int, default=None)
    pr.add_argument("--seed", type=int, default=0)
    pr.set_defaults(func=cmd_replan)

    ps = sub.add_parser("status", help="per-topic recall and exam readiness")
    ps.add_argument("--today", type=int, default=None)
    ps.set_defaults(func=cmd_status)

    pw = sub.add_parser("what-if", help="value of extra study time on one day")
    pw.add_argument("--day", type=int, required=True)
    pw.add_argument("--extra-minutes", type=int, required=True)
    pw.add_argument("--today", type=int, default=None)
    pw.add_argument("--seed", type=int, default=0)
    pw.set_defaults(func=cmd_what_if)

    pe = sub.add_parser("export", help="export the plan to CSV or Markdown")
    pe.add_argument("--format", choices=["csv", "md"], required=True)
    pe.add_argument("--out", required=True)
    pe.set_defaults(func=cmd_export)

    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
