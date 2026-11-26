"""ST4: fit retention on chronological held-out reviews.

Walk-forward (blocked, expanding-window) evaluation: each fold fits
on earlier blocks only and scores the next block one-step-ahead, so
no prediction ever sees its own outcome or any later event. The
baselines are the unfitted default, a naive constant, the fitted
exponential, and the pooled hierarchical model when data permits.
"""

import math

import pytest

from studyforge.evaluations import reliability_by_interval, walk_forward
from studyforge.memory import MIN_FIT_SAMPLES, fit_stability, initial_stability
from studyforge.models import get_model
from studyforge.models.base import HistoryEvent, _usable_pairs


def _exact_events(s_true, days):
    """Events whose recalls follow exp(-interval / s_true) exactly."""
    events = []
    prev = None
    for day in days:
        if prev is None:
            outcome = 1.0
        else:
            outcome = math.exp(-(day - prev) / s_true)
        events.append(HistoryEvent(day=day, outcome=outcome))
        prev = day
    return events


# Topic A: exact log-linear history, S_true = 2.0, six events.
TOPIC_A = ("A", 3, _exact_events(2.0, [0, 2, 5, 9, 14, 20]))
# Topic B: exact log-linear history, S_true = 4.0, four events.
TOPIC_B = ("B", 2, _exact_events(4.0, [0, 3, 7, 12]))
# Topic C: sparse, one usable pair only, so fits must fall back.
TOPIC_C = ("C", 3, _exact_events(2.0, [0, 4]))
# Topic D: binary right-or-wrong retrievals, for log loss and Brier.
TOPIC_D = ("D", 3, [
    HistoryEvent(day=0, outcome=1, binary=True),
    HistoryEvent(day=2, outcome=1, binary=True),
    HistoryEvent(day=4, outcome=0, binary=True),
    HistoryEvent(day=6, outcome=1, binary=True),
    HistoryEvent(day=8, outcome=0, binary=True),
])
# Topic E: one censored (unrecorded) outcome in the middle.
TOPIC_E = ("E", 3, [
    HistoryEvent(day=0, outcome=1.0),
    HistoryEvent(day=2, outcome=None),
    HistoryEvent(day=5, outcome=0.5),
    HistoryEvent(day=9, outcome=0.4),
])

TOPICS = [TOPIC_A, TOPIC_B, TOPIC_C, TOPIC_D, TOPIC_E]

MODELS = [
    get_model("default"),
    get_model("constant"),
    get_model("exponential"),
    get_model("hlr"),
]


def test_stage3_studyforge_4_registry_resolves_new_models():
    assert get_model("constant").version == "constant/1.0"
    assert get_model("default").version == "exponential-default/1.0"
    assert get_model("exponential").version == "exponential/1.0"
    assert get_model("hlr").version == "hlr/1.0"


def test_stage3_studyforge_4_default_never_fits():
    model = get_model("default")
    events = TOPIC_A[2]
    state = model.fit(events, difficulty=3)
    assert state.S == pytest.approx(initial_stability(3))
    assert state.last_review == 20
    # The default predicts from the difficulty stability, not a fit.
    assert model.predict(state, 2.0) == pytest.approx(math.exp(-2.0 / 1.0))


def test_stage3_studyforge_4_constant_predicts_train_mean():
    model = get_model("constant")
    events = TOPIC_A[2]
    state = model.fit(events, difficulty=3)
    mean = sum(e.outcome for e in events) / len(events)
    assert model.predict(state, 5.0) == pytest.approx(mean)
    # No recorded outcomes at all, but reviews exist: the naive
    # constant is 0.5, not a fitted value. A never-reviewed state
    # still predicts 0.0, matching the other models.
    no_outcomes = model.fit(
        [HistoryEvent(day=0), HistoryEvent(day=2)], difficulty=3)
    assert model.predict(no_outcomes, 5.0) == pytest.approx(0.5)
    never = model.fit([], difficulty=3)
    assert model.predict(never, 5.0) == pytest.approx(0.0)


def test_stage3_studyforge_4_walkforward_scores_each_event_once():
    report = walk_forward(MODELS, TOPICS, n_folds=4)
    by_model = {r["model"]: r for r in report["results"]}
    for version, result in by_model.items():
        # Every event with a recorded outcome is scored exactly once:
        # once per fold-block, never twice, never skipped.
        scored = result["n_scored"]
        total_outcomes = sum(
            1 for _, _, events in TOPICS
            for e in events if e.outcome is not None
        )
        assert scored == total_outcomes, version
        assert result["coverage"] == pytest.approx(1.0), version
        # The censored event is counted, never scored.
        assert result["n_censored"] == 1, version


def test_stage3_studyforge_4_folds_are_contiguous_time_blocks():
    report = walk_forward(MODELS, TOPICS, n_folds=4)
    assert report["n_folds"] == 4
    for result in report["results"]:
        by_topic: dict[str, list[dict]] = {}
        for fold in result["folds"]:
            by_topic.setdefault(fold["topic"], []).append(fold)
        for topic, folds in by_topic.items():
            ordered = sorted(folds, key=lambda f: f["fold"])
            boundaries = [f["boundary_day"] for f in ordered]
            assert boundaries == sorted(boundaries)
            assert len(set(boundaries)) == len(boundaries)
            # Each fold's test block follows its train window and
            # starts strictly later than the previous fold's.
            for fold in ordered:
                assert fold["n_train_events"] >= 0
                assert fold["n_test_events"] >= 1
                assert fold["min_test_day"] == fold["boundary_day"]
                if fold["max_train_day"] is not None:
                    assert fold["max_train_day"] < fold["boundary_day"]


def test_stage3_studyforge_4_prediction_uses_only_past_events():
    base = walk_forward(MODELS, [TOPIC_A], n_folds=4)
    flipped_topics = [("A", 3, [
        HistoryEvent(day=e.day, outcome=e.outcome)
        for e in TOPIC_A[2][:-1]
    ] + [HistoryEvent(day=20, outcome=0.9)])]
    flipped = walk_forward(MODELS, flipped_topics, n_folds=4)
    base_recs = {
        r["model"]: r for r in base["results"]
    }
    flip_recs = {
        r["model"]: r for r in flipped["results"]
    }
    for version in base_recs:
        # Records at or before the flipped day are identical: the
        # flipped outcome cannot reach an earlier prediction.
        b = [p for p in base_recs[version]["predictions"] if p["day"] <= 14]
        f = [p for p in flip_recs[version]["predictions"] if p["day"] <= 14]
        assert b == f, version


def test_stage3_studyforge_4_fit_never_uses_future_events():
    report = walk_forward(
        [get_model("exponential")], [TOPIC_A, TOPIC_B], n_folds=4)
    result = report["results"][0]
    events_by_topic = {t: e for t, _, e in [TOPIC_A, TOPIC_B]}
    for fit in result["fits"]:
        topic = fit["topic"]
        events = events_by_topic[topic]
        train = [e for e in events if e.day < fit["boundary_day"]]
        expected = fit_stability(_usable_pairs(train))
        if expected is None:
            assert fit["fallback"] is True
            assert fit["S"] == pytest.approx(
                initial_stability(3 if topic == "A" else 2))
        else:
            assert fit["fallback"] is False
            assert fit["S"] == pytest.approx(expected)


def test_stage3_studyforge_4_exact_recovery_oracle():
    report = walk_forward(
        [get_model("exponential")], [TOPIC_A], n_folds=4)
    result = report["results"][0]
    # Fold 3 trains on days [0, 2, 5, 9]: three exact pairs, so the
    # fit recovers the true stability of 2.0 days exactly.
    fold3 = [f for f in result["fits"] if f["fold"] == 3][0]
    assert fold3["S"] == pytest.approx(2.0, abs=1e-9)
    assert fold3["fallback"] is False
    # Its first prediction is therefore exact: 5 days after the last
    # train review, exp(-5/2) against the observed exp(-5/2).
    first = [p for p in result["predictions"]
             if p["fold"] == 3][0]
    assert first["predicted"] == pytest.approx(first["observed"], abs=1e-9)
    assert first["predicted"] == pytest.approx(math.exp(-2.5), abs=1e-9)


def test_stage3_studyforge_4_baselines_all_reported():
    report = walk_forward(MODELS, TOPICS, n_folds=4)
    versions = [r["model"] for r in report["results"]]
    assert versions == [
        "exponential-default/1.0", "constant/1.0",
        "exponential/1.0", "hlr/1.0",
    ]
    for result in report["results"]:
        # Every baseline reports metrics with uncertainty, and the
        # report names its sample size, coverage, and censoring.
        assert result["n_scored"] > 0
        assert "mae" in result and result["mae"] is not None
        assert result["mae_ci"] is not None
        assert result["n_events"] > 0
        assert result["coverage"] == pytest.approx(1.0)
        assert result["n_censored"] == 1
        assert result["reliability_by_interval"]
        assert result["by_topic"]
        assert result["leakage_checks"]


def test_stage3_studyforge_4_too_little_data_falls_back():
    report = walk_forward(MODELS, [TOPIC_C], n_folds=4)
    for result in report["results"]:
        # Topic C has one usable pair at most, below the minimum, so
        # every fitted fold falls back instead of inventing a fit.
        # The unfitted default never falls back: it is unfit by
        # design, so its fallback count stays zero.
        if result["model"] == "exponential-default/1.0":
            assert result["fallbacks"] == 0
            assert all(
                f["fallback_reason"].startswith("unfitted by design")
                for f in result["fits"]
            )
        else:
            assert result["fallbacks"] >= 1
            assert any(f["fallback"] for f in result["fits"])
        assert result["n_scored"] == 2
    constant = [r for r in report["results"]
                if r["model"] == "constant/1.0"][0]
    # Fold 0 has no train history: the constant falls back to 0.5,
    # and the never-reviewed first event still predicts 0.0.
    assert constant["fits"][0]["constant"] == pytest.approx(0.5)
    assert constant["predictions"][0]["predicted"] == pytest.approx(0.0)
    # Fold 1 trains on the single recorded outcome 1.0, so the
    # constant for the next event is 1.0.
    assert constant["fits"][1]["constant"] == pytest.approx(1.0)
    assert constant["predictions"][1]["predicted"] == pytest.approx(1.0)


def test_stage3_studyforge_4_reliability_by_interval_partitions_records():
    report = walk_forward(
        [get_model("exponential")], [TOPIC_A, TOPIC_B], n_folds=4)
    result = report["results"][0]
    buckets = result["reliability_by_interval"]
    assert sum(b["n"] for b in buckets) == result["n_scored"]
    # Hand-check: the 4-7 day bucket holds the 5-day gaps, each
    # predicted from the fitted stability of its fold.
    mid = [b for b in buckets if b["label"] == "4-7d"][0]
    assert mid["n"] >= 1
    assert 0.0 <= mid["observed"] <= 1.0
    assert mid["mae"] is not None
    # The helper itself partitions any record list without dropping.
    again = reliability_by_interval(result["predictions"])
    assert sum(b["n"] for b in again) == result["n_scored"]


def test_stage3_studyforge_4_leakage_checks_pass():
    report = walk_forward(MODELS, TOPICS, n_folds=4)
    for result in report["results"]:
        assert result["leakage_checks"], result["model"]
        for check in result["leakage_checks"]:
            assert check["passed"] is True, (result["model"], check)


def test_stage3_studyforge_4_censoring_counted_not_scored():
    report = walk_forward(
        [get_model("exponential")], [TOPIC_E], n_folds=4)
    result = report["results"][0]
    assert result["n_censored"] == 1
    assert result["n_scored"] == 3
    assert all(p["observed"] is not None for p in result["predictions"])
    assert all(p["day"] != 2 for p in result["predictions"])


def test_stage3_studyforge_4_walkforward_is_deterministic():
    first = walk_forward(MODELS, TOPICS, n_folds=4)
    second = walk_forward(MODELS, TOPICS, n_folds=4)
    assert first == second


def test_stage3_studyforge_4_binary_outcomes_score_log_loss_and_brier():
    report = walk_forward(
        [get_model("exponential"), get_model("constant")],
        [TOPIC_D], n_folds=4)
    for result in report["results"]:
        assert result["n_binary"] == 5
        assert result["log_loss"] is not None
        assert result["brier"] is not None
        assert result["log_loss_ci"] is not None
        assert 0.0 <= result["brier"] <= 1.0


def test_stage3_studyforge_4_min_fit_samples_guard():
    # Fewer than MIN_FIT_SAMPLES usable pairs: the fitted model
    # reports the fallback, it never fits on two points and calls
    # it a personal stability.
    assert MIN_FIT_SAMPLES == 3
    report = walk_forward(
        [get_model("exponential")], [TOPIC_C], n_folds=4)
    result = report["results"][0]
    for fit in result["fits"]:
        assert fit["n_train_pairs"] < MIN_FIT_SAMPLES
        assert fit["fallback"] is True
