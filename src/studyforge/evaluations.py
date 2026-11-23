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
"""

from __future__ import annotations

import math

from .models.base import HistoryEvent, ModelProtocol, normalize_outcome

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
