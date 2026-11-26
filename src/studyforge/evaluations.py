"""Held-out tournament scoring for the versioned memory models.

A tournament splits each topic's logged history at a cutoff day: the
model fits on events strictly before the cutoff and is scored
one-step-ahead on the events from the cutoff on, in day order, with
the state rolled forward through each observed outcome after it is
scored. Nothing after the cutoff influences the fit, and no train
event is ever scored, so the numbers compare models on equal footing.

Metrics
-------
Binary outcomes (right-or-wrong retrievals) score by log loss and
Brier score. Fractional outcomes (fraction remembered) score by mean
absolute error and mean squared error. Every metric is reported with
a 95% confidence interval: a normal approximation (mean +/- 1.96
standard errors) for the metric means, and a Wilson score interval
for the observed rate in each calibration bin.

Calibration data is returned as plain JSON-able records (bin edges,
prediction means, observed rates, counts, Wilson intervals), so the
dashboard can plot it without recomputing anything.

A model that cannot run on a topic's data is skipped for that topic
with a recorded reason instead of silently scoring zero: the FSRS
adapter needs item-level binary outcomes, so on topic-level
fractional history it sits that topic out and says why. A model that
can handle none of the topics is excluded outright. The default
stays the simple model until held-out evidence supports switching; a
tournament result is evidence, not a switch.

Walk-forward evaluation
------------------------
A tournament scores one cutoff. Walk-forward scores every event
exactly once, in chronological blocks: each topic's history splits
into contiguous time blocks, and each block is scored with a model
fitted on the earlier blocks only, one-step-ahead in day order. The
train window grows as the folds advance, so later folds fit on more
history and every prediction uses only past events. The report names
its sample size, coverage (scored over scoreable held-out events),
censoring (held-out events with no recorded outcome, counted and
never scored as zero), stratification (per topic and per interval
since the last review), uncertainty (the same 95% intervals as the
tournament), and leakage checks (train days strictly before each
fold boundary, test days from it, and the fitted state's last review
matching the train window). A fold whose train window is too small
to fit falls back to the default stability, or to 0.5 for the naive
constant, and the report says so instead of inventing a fit.
"""

from __future__ import annotations

import math

from .memory import MIN_FIT_SAMPLES
from .models.base import (
    HistoryEvent,
    ModelProtocol,
    _usable_pairs,
    normalize_outcome,
)
from .models.hlr import HierarchicalHalfLifeModel

_EPS = 1e-15
_Z_95 = 1.96


def _check_scored(preds, outcomes, binary: bool) -> None:
    if len(preds) != len(outcomes):
        raise ValueError(
            f"need one prediction per outcome, got {len(preds)} "
            f"predictions and {len(outcomes)} outcomes"
        )
    if not preds:
        raise ValueError("need at least one scored event")
    for p in preds:
        if not math.isfinite(p) or not 0.0 <= p <= 1.0:
            raise ValueError(
                f"predictions must be probabilities between 0 and 1, "
                f"got {p!r}"
            )
    for o in outcomes:
        norm = normalize_outcome(o)
        if binary and norm.kind != "binary":
            raise ValueError(
                f"binary scoring needs outcomes of exactly 0 or 1, got "
                f"{o!r}"
            )
        if not binary and norm.kind == "missing":
            raise ValueError("cannot score a missing outcome")


def log_loss(preds: list[float], outcomes: list[float]) -> float:
    """Mean negative log likelihood of binary outcomes (nats)."""
    _check_scored(preds, outcomes, binary=True)
    total = 0.0
    for p, o in zip(preds, outcomes):
        pc = min(1.0 - _EPS, max(_EPS, p))
        total += -(o * math.log(pc) + (1.0 - o) * math.log(1.0 - pc))
    return total / len(preds)


def brier_score(preds: list[float], outcomes: list[float]) -> float:
    """Mean squared error of predicted probabilities, binary outcomes."""
    _check_scored(preds, outcomes, binary=True)
    return sum((p - o) ** 2 for p, o in zip(preds, outcomes)) / len(preds)


def mae(preds: list[float], outcomes: list[float]) -> float:
    """Mean absolute error, for fractional responses."""
    _check_scored(preds, outcomes, binary=False)
    return sum(abs(p - o) for p, o in zip(preds, outcomes)) / len(preds)


def mse(preds: list[float], outcomes: list[float]) -> float:
    """Mean squared error, for fractional responses."""
    _check_scored(preds, outcomes, binary=False)
    return sum((p - o) ** 2 for p, o in zip(preds, outcomes)) / len(preds)


def mean_ci(values: list[float]) -> tuple[float, float, float]:
    """Mean with a 95% normal-approximation confidence interval.

    Returns (mean, lo, hi) unclipped: log loss is unbounded above,
    so callers clip when their metric lives in 0 to 1. With one
    value the interval is a point; with none it raises ValueError.
    """
    if not values:
        raise ValueError("need at least one value for a confidence interval")
    n = len(values)
    mean = sum(values) / n
    if n == 1:
        return mean, mean, mean
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    half = _Z_95 * math.sqrt(var / n)
    return mean, mean - half, mean + half


def wilson_ci(k: int, n: int) -> tuple[float, float]:
    """Wilson score 95% interval for k successes in n trials."""
    if n <= 0:
        raise ValueError("need at least one trial for a Wilson interval")
    if not 0 <= k <= n:
        raise ValueError(f"successes {k} must be between 0 and {n}")
    z2 = _Z_95 ** 2
    p = k / n
    center = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = _Z_95 * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return max(0.0, center - half), min(1.0, center + half)


def calibration_bins(
    preds: list[float], outcomes: list[float], n_bins: int = 10
) -> list[dict]:
    """Calibration data: one record per predicted-probability bin.

    Bins are equal-width over 0 to 1. Each record carries the bin
    edges, the count, the mean predicted and observed values, and the
    Wilson 95% interval for the observed rate (from the count of
    outcomes above 0.5, so fractional outcomes contribute their
    rounded direction). Empty bins report None means and a 0 to 1
    interval instead of invented numbers.
    """
    if len(preds) != len(outcomes):
        raise ValueError(
            f"need one prediction per outcome, got {len(preds)} "
            f"predictions and {len(outcomes)} outcomes"
        )
    if n_bins <= 0:
        raise ValueError(f"need at least one bin, got {n_bins}")
    bins: list[dict] = []
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        idx = [j for j, p in enumerate(preds)
               if (lo <= p < hi) or (i == n_bins - 1 and p == 1.0)]
        n = len(idx)
        if n == 0:
            bins.append({"bin": i, "lo": lo, "hi": hi, "n": 0,
                         "predicted": None, "observed": None,
                         "ci_lo": 0.0, "ci_hi": 1.0})
            continue
        mean_p = sum(preds[j] for j in idx) / n
        mean_o = sum(outcomes[j] for j in idx) / n
        k = sum(1 for j in idx if outcomes[j] > 0.5)
        ci_lo, ci_hi = wilson_ci(k, n)
        bins.append({"bin": i, "lo": lo, "hi": hi, "n": n,
                     "predicted": mean_p, "observed": mean_o,
                     "ci_lo": ci_lo, "ci_hi": ci_hi})
    return bins


def _split(events: list[HistoryEvent], cutoff: int):
    """(train, test) split at the cutoff: train strictly before it."""
    ordered = sorted(events, key=lambda e: e.day)
    return ([e for e in ordered if e.day < cutoff],
            [e for e in ordered if e.day >= cutoff])


def _score_topic(model: ModelProtocol, events: list[HistoryEvent],
                 cutoff: int, difficulty: int) -> list[dict]:
    """One-step-ahead scored predictions for one topic's held-out events.

    The model fits on pre-cutoff events only. Each held-out event is
    predicted from the state as of its day, then the state rolls
    forward through the observed outcome. Events with no recorded
    outcome are skipped, never scored as 0. Each record is tagged
    with the model version.
    """
    train, test = _split(events, cutoff)
    state = model.fit(train, as_of=cutoff, difficulty=difficulty)
    records: list[dict] = []
    for ev in test:
        if ev.outcome is None:
            continue
        norm = normalize_outcome(ev.outcome)
        if state.last_review is None:
            predicted = 0.0
        else:
            predicted = model.predict(state, ev.day - state.last_review)
        records.append({
            "model": model.version,
            "day": ev.day,
            "predicted": predicted,
            "observed": norm.value,
            "kind": "binary" if (ev.binary or norm.kind == "binary")
                    else "fractional",
        })
        state = model.transition(state, ev.day, outcome=ev.outcome)
    return records


def _summarize(records: list[dict]) -> dict:
    """Metric summary for one model's scored records."""
    binary = [r for r in records if r["kind"] == "binary"]
    fractional = [r for r in records if r["kind"] == "fractional"]
    out: dict = {
        "n_binary": len(binary),
        "n_fractional": len(fractional),
        "predictions": records,
        "calibration": calibration_bins(
            [r["predicted"] for r in records],
            [r["observed"] for r in records]),
    }
    if binary:
        bp = [r["predicted"] for r in binary]
        bo = [r["observed"] for r in binary]
        ll_terms = []
        br_terms = []
        for p, o in zip(bp, bo):
            pc = min(1.0 - _EPS, max(_EPS, p))
            ll_terms.append(
                -(o * math.log(pc) + (1.0 - o) * math.log(1.0 - pc)))
            br_terms.append((p - o) ** 2)
        out["log_loss"] = sum(ll_terms) / len(ll_terms)
        out["log_loss_ci"] = mean_ci(ll_terms)[1:]
        out["brier"] = sum(br_terms) / len(br_terms)
        out["brier_ci"] = mean_ci(br_terms)[1:]
    else:
        out["log_loss"] = None
        out["log_loss_ci"] = None
        out["brier"] = None
        out["brier_ci"] = None
    if fractional:
        fp = [r["predicted"] for r in fractional]
        fo = [r["observed"] for r in fractional]
        ae_terms = [abs(p - o) for p, o in zip(fp, fo)]
        se_terms = [(p - o) ** 2 for p, o in zip(fp, fo)]
        out["mae"] = sum(ae_terms) / len(ae_terms)
        out["mae_ci"] = mean_ci(ae_terms)[1:]
        out["mse"] = sum(se_terms) / len(se_terms)
        out["mse_ci"] = mean_ci(se_terms)[1:]
    else:
        out["mae"] = None
        out["mae_ci"] = None
        out["mse"] = None
        out["mse_ci"] = None
    return out


def tournament(
    models: list[ModelProtocol],
    topics: list[tuple[str, int, list[HistoryEvent]]],
    cutoff: int,
) -> dict:
    """Score each model on every topic's held-out history.

    ``topics`` is a list of (topic id, difficulty, history) tuples.
    A model scores the topics it can handle: a topic whose history
    a model cannot fit (the FSRS adapter on topic-level fractional
    recall, for example) is skipped for that model, and the result
    records which topics were scored. A model that can handle none
    of the topics is excluded from the results with the reason
    recorded. Results run in the order the models were given, so
    repeated runs are identical.
    """
    if cutoff < 0:
        raise ValueError(f"cutoff must be 0 or later, got {cutoff}")
    results: list[dict] = []
    excluded: list[dict] = []
    for model in models:
        records: list[dict] = []
        scored: list[str] = []
        skipped: list[dict] = []
        for tid, difficulty, events in topics:
            try:
                topic_records = _score_topic(model, events, cutoff,
                                             difficulty)
            except ValueError as e:
                skipped.append({"topic": tid, "reason": str(e)})
                continue
            records.extend(topic_records)
            scored.append(tid)
        if not scored:
            reason = (skipped[0]["reason"] if skipped else
                      "no topics with held-out events")
            excluded.append({"model": model.version, "reason": reason})
            continue
        summary = _summarize(records)
        summary["model"] = model.version
        summary["topics"] = scored
        summary["skipped"] = skipped
        results.append(summary)
    return {"cutoff": cutoff, "results": results, "excluded": excluded}


# ----------------------------------------------------------------------
# Walk-forward evaluation
# ----------------------------------------------------------------------

_INTERVAL_EDGES = (1, 3, 7, 14)


def _fold_starts(
    events: list[HistoryEvent], n_folds: int
) -> list[int]:
    """Start indices of the walk-forward test blocks, day-aligned.

    The ordered events split into ``n_folds`` contiguous blocks of
    near-equal size; each block after the first is scored once, with
    all earlier blocks as its train window, and the first block is
    scored with an empty train window, so the too-little-data
    fallback is exercised instead of assumed. A boundary never
    splits a day: it moves forward to the next event on a later
    day, so every train event strictly precedes every test event
    in its fold. Returns the start index of each block, the first
    always 0.
    """
    if n_folds < 1:
        raise ValueError(f"need at least one fold, got {n_folds}")
    ordered = sorted(events, key=lambda e: e.day)
    n = len(ordered)
    if n == 0:
        return []
    starts = [0]
    for k in range(1, n_folds):
        idx = (n * k) // n_folds
        while 0 < idx < n and ordered[idx].day == ordered[idx - 1].day:
            idx += 1
        if 0 < idx < n and idx not in starts:
            starts.append(idx)
    return starts


def _prepare_for_fold(
    model: ModelProtocol, all_histories: list[list[HistoryEvent]],
    boundary_day: int,
) -> ModelProtocol:
    """A fresh model instance for one fold, fitted on past only.

    Most models are stateless across folds and pass through
    unchanged. The hierarchical model carries a pooled value, so
    each fold gets a fresh instance pooled from every topic's
    history strictly before the fold boundary.
    """
    if isinstance(model, HierarchicalHalfLifeModel):
        return HierarchicalHalfLifeModel.fit_pooled(
            all_histories, as_of=boundary_day)
    return model


def _score_fold(
    model: ModelProtocol, train: list[HistoryEvent],
    test: list[HistoryEvent], difficulty: int, fold: int, topic: str,
) -> tuple[object, list[dict], int]:
    """One-step-ahead scored predictions for one fold's test block.

    The model fits on the train window only. Each test event is
    predicted from the state as of its day, then the state rolls
    forward through the observed outcome, so a prediction never
    uses its own outcome. Events with no recorded outcome are
    censored: counted, never scored. Returns the fitted state, the
    scored records, and the censored count.
    """
    state = model.fit(train, as_of=None, difficulty=difficulty)
    fitted_state = state
    records: list[dict] = []
    n_censored = 0
    for ev in test:
        if ev.outcome is None:
            n_censored += 1
            continue
        norm = normalize_outcome(ev.outcome)
        if state.last_review is None:
            predicted, elapsed = 0.0, None
        else:
            elapsed = ev.day - state.last_review
            predicted = model.predict(state, elapsed)
        records.append({
            "model": model.version,
            "topic": topic,
            "fold": fold,
            "day": ev.day,
            "elapsed": elapsed,
            "predicted": predicted,
            "observed": norm.value,
            "kind": "binary" if (ev.binary or norm.kind == "binary")
                    else "fractional",
        })
        state = model.transition(state, ev.day, outcome=ev.outcome)
    return fitted_state, records, n_censored


def _fold_fallback(
    model: ModelProtocol, train: list[HistoryEvent], difficulty: int,
) -> tuple[bool, str]:
    """Whether one fold's fit fell back for lack of data, and why.

    The fitted exponential and the hierarchical model need
    MIN_FIT_SAMPLES usable pairs before a per-topic fit is trusted;
    with fewer, the default or pooled stability stands. The naive
    constant needs at least one recorded outcome, else it is 0.5.
    The unfitted default never falls back: it is unfit by design.
    """
    version = model.version
    if version == "exponential-default/1.0":
        return False, "unfitted by design; the default never fits"
    if version == "constant/1.0":
        outcomes = [e.outcome for e in train if e.outcome is not None]
        if not outcomes:
            return True, (
                "no recorded outcomes in the train window; "
                "constant is 0.5"
            )
        return False, ""
    pairs = _usable_pairs(train)
    if len(pairs) < MIN_FIT_SAMPLES:
        return True, (
            f"only {len(pairs)} usable pairs, fewer than "
            f"{MIN_FIT_SAMPLES}; default stability stands"
        )
    return False, ""


def _score_topic_folds(
    model: ModelProtocol, topic: str, difficulty: int,
    events: list[HistoryEvent],
    all_histories: list[list[HistoryEvent]], n_folds: int,
) -> tuple[list[dict], list[dict], list[dict], int]:
    """All folds for one topic: records, fits, fold summaries, censored.

    Every event lands in exactly one test block and is scored at
    most once. A model that cannot fit the topic raises ValueError
    and the topic is skipped whole, with no partial records kept.
    """
    ordered = sorted(events, key=lambda e: e.day)
    starts = _fold_starts(ordered, n_folds)
    records: list[dict] = []
    fits: list[dict] = []
    folds: list[dict] = []
    n_censored = 0
    for fold_idx, start in enumerate(starts):
        end = (starts[fold_idx + 1] if fold_idx + 1 < len(starts)
               else len(ordered))
        train = ordered[:start]
        test = ordered[start:end]
        boundary_day = test[0].day
        fold_model = _prepare_for_fold(model, all_histories, boundary_day)
        fitted_state, fold_records, fold_censored = _score_fold(
            fold_model, train, test, difficulty, fold_idx, topic)
        fell_back, reason = _fold_fallback(fold_model, train, difficulty)
        fits.append({
            "topic": topic,
            "fold": fold_idx,
            "boundary_day": boundary_day,
            "n_train_pairs": len(_usable_pairs(train)),
            "fallback": fell_back,
            "fallback_reason": reason,
            "S": (None if fold_model.version == "constant/1.0"
                  else float(fitted_state.S)),
            "constant": (
                fold_model.fitted_constant
                if fold_model.version == "constant/1.0" else None
            ),
            "last_review": fitted_state.last_review,
        })
        summary = {
            "topic": topic,
            "fold": fold_idx,
            "boundary_day": boundary_day,
            "max_train_day": train[-1].day if train else None,
            "min_test_day": test[0].day,
            "n_train_events": len(train),
            "n_train_pairs": len(_usable_pairs(train)),
            "n_test_events": len(test),
            "n_scored": len(fold_records),
            "n_censored": fold_censored,
            "fallback": fell_back,
            "fallback_reason": reason,
        }
        for key in ("log_loss", "brier", "mae", "mse", "log_loss_ci",
                    "brier_ci", "mae_ci", "mse_ci", "n_binary",
                    "n_fractional"):
            summary[key] = None
        if fold_records:
            fold_summary = _summarize(fold_records)
            for key in ("log_loss", "brier", "mae", "mse", "log_loss_ci",
                        "brier_ci", "mae_ci", "mse_ci", "n_binary",
                        "n_fractional"):
                summary[key] = fold_summary[key]
        records.extend(fold_records)
        folds.append(summary)
        n_censored += fold_censored
    return records, fits, folds, n_censored


def reliability_by_interval(
    records: list[dict], edges: tuple[int, ...] = _INTERVAL_EDGES,
) -> list[dict]:
    """Reliability by days since the last review.

    Scored records bucket by elapsed days: no prior review, then
    the ranges the edges cut (0-1, 2-3, 4-7, 8-14, 15+ days by
    default). Each bucket reports its count, mean predicted and
    observed recall, mean absolute error, and log loss and Brier
    score when it holds binary records. Empty buckets report None
    instead of invented numbers.
    """
    bounds: list[tuple[int | None, int | None, str]] = [
        (None, None, "no prior review")]
    lo = 0
    for edge in edges:
        bounds.append((lo, edge, f"{lo}-{edge}d"))
        lo = edge + 1
    bounds.append((lo, None, f"{lo}+d"))

    def _bucket(elapsed: int | None) -> int:
        if elapsed is None:
            return 0
        for i, (blo, bhi, _) in enumerate(bounds[1:], start=1):
            if bhi is not None and elapsed <= bhi:
                return i
        return len(bounds) - 1

    grouped: list[list[dict]] = [[] for _ in bounds]
    for record in records:
        grouped[_bucket(record.get("elapsed"))].append(record)
    out: list[dict] = []
    for (_, _, label), bucket in zip(bounds, grouped):
        n = len(bucket)
        if n == 0:
            out.append({"label": label, "n": 0, "predicted": None,
                        "observed": None, "mae": None,
                        "log_loss": None, "brier": None})
            continue
        preds = [r["predicted"] for r in bucket]
        obs = [r["observed"] for r in bucket]
        entry: dict = {
            "label": label,
            "n": n,
            "predicted": sum(preds) / n,
            "observed": sum(obs) / n,
            "mae": sum(abs(p - o) for p, o in zip(preds, obs)) / n,
            "log_loss": None,
            "brier": None,
        }
        binary = [r for r in bucket if r["kind"] == "binary"]
        if binary:
            bp = [r["predicted"] for r in binary]
            bo = [r["observed"] for r in binary]
            entry["log_loss"] = log_loss(bp, bo)
            entry["brier"] = brier_score(bp, bo)
        out.append(entry)
    return out


def walk_forward(
    models: list[ModelProtocol],
    topics: list[tuple[str, int, list[HistoryEvent]]],
    n_folds: int = 4,
) -> dict:
    """Score each model by expanding-window walk-forward folds.

    ``topics`` is a list of (topic id, difficulty, history) tuples.
    Each topic's history splits into contiguous time blocks; every
    block is scored once, fitted on the earlier blocks only, so
    every held-out event is scored exactly once and no prediction
    ever sees its own outcome or a later event. The baselines to
    compare are the caller's models: the unfitted default, the
    naive constant, the fitted exponential, and the pooled
    hierarchical model when the data permits it.

    Each model's result reports its sample size (events, scored,
    censored), coverage (scored over scoreable held-out events),
    the tournament metrics with 95% confidence intervals,
    calibration bins, reliability by interval since the last
    review, a per-topic breakdown, per-fold and per-fit records
    naming every fallback, and leakage checks verifying the train
    window precedes each fold boundary and the fitted state
    matches it. A model that scores no topic is excluded with the
    reason recorded. Results run in the order the models were
    given, so repeated runs are identical.
    """
    if n_folds < 1:
        raise ValueError(f"need at least one fold, got {n_folds}")
    if not models:
        raise ValueError("need at least one model")
    all_histories = [events for _, _, events in topics]
    results: list[dict] = []
    excluded: list[dict] = []
    for model in models:
        records: list[dict] = []
        fits: list[dict] = []
        folds: list[dict] = []
        skipped: list[dict] = []
        n_events = 0
        n_censored = 0
        for tid, difficulty, events in topics:
            n_events += len(events)
            if not events:
                skipped.append({"topic": tid, "reason": "no logged events"})
                continue
            try:
                (t_records, t_fits, t_folds,
                 t_censored) = _score_topic_folds(
                    model, tid, difficulty, events, all_histories, n_folds)
            except ValueError as e:
                skipped.append({"topic": tid, "reason": str(e)})
                continue
            records.extend(t_records)
            fits.extend(t_fits)
            folds.extend(t_folds)
            n_censored += t_censored
        n_scoreable = sum(
            1 for _, _, evs in topics
            for e in evs if e.outcome is not None
        )
        if not records:
            reason = (skipped[0]["reason"] if skipped else
                      "no topics with held-out events")
            excluded.append({"model": model.version, "reason": reason})
            continue
        summary = _summarize(records)
        summary["model"] = model.version
        summary["topics"] = [
            tid for tid, _, _ in topics
            if tid not in {s["topic"] for s in skipped}
        ]
        summary["skipped"] = skipped
        summary["n_folds"] = n_folds
        summary["n_events"] = n_events
        summary["n_scored"] = len(records)
        summary["n_censored"] = n_censored
        summary["coverage"] = (
            len(records) / n_scoreable if n_scoreable else 0.0
        )
        summary["reliability_by_interval"] = reliability_by_interval(
            records)
        by_topic: dict[str, dict] = {}
        for tid, _, _ in topics:
            topic_records = [r for r in records if r["topic"] == tid]
            entry: dict = {"n_scored": len(topic_records)}
            for key in ("log_loss", "brier", "mae", "mse",
                        "log_loss_ci", "brier_ci", "mae_ci", "mse_ci",
                        "n_binary", "n_fractional"):
                entry[key] = None
            if topic_records:
                topic_summary = _summarize(topic_records)
                for key in ("log_loss", "brier", "mae", "mse",
                            "log_loss_ci", "brier_ci", "mae_ci",
                            "mse_ci", "n_binary", "n_fractional"):
                    entry[key] = topic_summary[key]
            by_topic[tid] = entry
        summary["by_topic"] = by_topic
        checks: list[dict] = []
        for name, passed, detail in _leakage_checks(fits, folds):
            checks.append({"check": name, "passed": passed,
                           "detail": detail})
        summary["leakage_checks"] = checks
        summary["fits"] = fits
        summary["folds"] = folds
        summary["fallbacks"] = sum(1 for f in fits if f["fallback"])
        results.append(summary)
    return {"n_folds": n_folds,
            "topics": [tid for tid, _, _ in topics],
            "results": results, "excluded": excluded}


def _leakage_checks(
    fits: list[dict], folds: list[dict]
) -> list[tuple[str, bool, str]]:
    """Data-level leakage checks over every scored fold.

    Each check aggregates over all folds and passes only when every
    fold passes it: the train window's days all precede the fold
    boundary, the test block starts at the boundary, and the fitted
    state's last review matches the train window (or is absent
    when the train window is empty). These are computed from the
    recorded folds, not asserted and forgotten.
    """
    fold_by_key = {(f["topic"], f["fold"]): f for f in folds}
    train_before = True
    test_from = True
    fit_matches = True
    checked = 0
    for fit in fits:
        key = (fit["topic"], fit["fold"])
        fold = fold_by_key[key]
        checked += 1
        # Every train day precedes the fold boundary, so the fit
        # never sees the test block or any later event.
        if fold["max_train_day"] is not None and not (
                fold["max_train_day"] < fold["boundary_day"]):
            train_before = False
        # The test block starts exactly at the boundary.
        if fold["min_test_day"] != fold["boundary_day"]:
            test_from = False
        # The fitted state's last review is the train window's last
        # day, or None for an empty train window.
        if fit["last_review"] != fold["max_train_day"]:
            fit_matches = False
    return [
        ("train_before_fold", train_before,
         f"{checked} folds checked; every train window precedes its "
         "fold boundary"),
        ("test_from_fold", test_from,
         f"{checked} folds checked; every test block starts at its "
         "fold boundary"),
        ("fit_last_review_matches_train", fit_matches,
         f"{checked} fits checked; every fitted state matches its "
         "train window"),
    ]
