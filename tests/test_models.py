"""A06 regression fixture: versioned model interface and baseline tournament.

Written against the defects named in handoffs section 151 BEFORE the
fix, so each test fails on the pre-fix code for the stated reason:

* there is no model interface: planner.simulate computes the
  exponential update inline, so no other memory model can be fitted,
  compared, or swapped in.
* fit_stability is a per-topic log-linear regression with no pooling,
  so sparse topics get noisy or default stabilities.
* there is no held-out scoring: nothing reports log loss, Brier
  score, absolute or squared error, calibration, or confidence
  intervals for any model.
* FSRS stability (time to 90% retrievability) and StudyForge
  stability (exponential e-folding time) share no conversion, so the
  two units can be silently interchanged.

Hand-derived scenario constants (memory.py: S0 = 3/difficulty; a
review multiplies S by 1 + (1 - R); R = exp(-t/S) at the review, 0
for the first review). Difficulty 3, so S0 = 1.0; reviews on days 3
and 10 with no observed recall:

    after day 3:  S = 1.0 * 2 = 2.0, last_review = 3
    day 10:       R = exp(-7/2) = 0.030197, S = 2 * (1 + 1 - R)
                  = 3.939605, last_review = 10
    recall at day 7 from the day-3 state: exp(-(7-3)/2) = exp(-2)
                  = 0.135335

HLR shrinkage constants (k = 4 pseudo-observations): topic A pairs
from true S = 2.0 give a per-topic rate u = 0.5; topic B pairs from
true S = 4.0 pooled with A give u_pool = 0.375, so the shrunk rate
is (3 * 0.5 + 4 * 0.375) / 7 = 3/7 and the shrunk S is 7/3.

FSRS constants: R(t) = (1 + 19/81 * t / S)^(-1/2), so R = 0.9
exactly at t = S; the unit conversion is S_fsrs = S_sf * ln(10/9).
"""

import json
import math

import pytest

from studyforge import cli as _cli
from studyforge import dashboard as dash
from studyforge import state as st
from studyforge.memory import TopicState
from studyforge.planner import PlanTopic, build_plan, simulate

# The new modules do not exist yet: importing them is itself the
# failing fixture for the interface half of this unit.
from studyforge.models import (
    FSRSAdapter,
    HierarchicalHalfLifeModel,
    ExponentialModel,
    HistoryEvent,
    ModelState,
    get_model,
    load_model,
    normalize_outcome,
)
from studyforge.models.fsrs_adapter import (
    FSRS_DECAY,
    FSRS_FACTOR,
    from_studyforge_stability,
    to_studyforge_stability,
)
from studyforge.evaluations import (
    brier_score,
    calibration_bins,
    log_loss,
    mae,
    mean_ci,
    mse,
    tournament,
    wilson_ci,
)

S_AFTER_DAY3 = 2.0
R_AT_DAY10 = math.exp(-7.0 / S_AFTER_DAY3)
S_AFTER_DAY10 = S_AFTER_DAY3 * (1.0 + (1.0 - R_AT_DAY10))
RECALL_DAY7 = math.exp(-2.0)
LN_10_OVER_9 = math.log(10.0 / 9.0)


def _events(days_outcomes):
    """HistoryEvent list from (day, outcome) pairs, fractional kind."""
    return [HistoryEvent(day=d, outcome=o) for d, o in days_outcomes]


def _topic_a_events():
    """Topic A reviews, no observed recall (pure replay)."""
    return _events([(3, None), (10, None)])


def _hlr_pairs_events():
    """Topic A (true S 2.0) and topic B (true S 4.0) interval/recall pairs.

    Each event after the first carries the recall observed at that
    review; the interval to the previous review is what the fit
    consumes. The first event anchors the intervals and yields no
    pair itself, matching how the log command records intervals.
    Topic A pairs: (2, exp(-1)), (4, exp(-2)), (1, exp(-0.5)).
    Topic B pairs: (2, exp(-0.5)), (4, exp(-1)), (1, exp(-0.25)).
    """
    return (
        [HistoryEvent(day=1, outcome=0.9),
         HistoryEvent(day=3, outcome=math.exp(-1.0)),
         HistoryEvent(day=7, outcome=math.exp(-2.0)),
         HistoryEvent(day=8, outcome=math.exp(-0.5))],
        [HistoryEvent(day=1, outcome=0.9),
         HistoryEvent(day=3, outcome=math.exp(-0.5)),
         HistoryEvent(day=7, outcome=math.exp(-1.0)),
         HistoryEvent(day=8, outcome=math.exp(-0.25))],
    )


# --- Exponential model: the unchanged baseline, now behind the interface ---


def test_stage3_studyforge_06_exponential_reproduces_topic_state():
    model = ExponentialModel()
    assert model.version == "exponential/1.0"
    state = ModelState(S=1.0, last_review=None, difficulty=3)
    assert model.predict(state, 5.0) == 0.0  # never reviewed: nothing learned
    state = model.transition(state, 3)
    assert state.S == pytest.approx(S_AFTER_DAY3)
    assert state.last_review == 3
    assert model.predict(state, 4.0) == pytest.approx(RECALL_DAY7)
    state = model.transition(state, 10)
    assert state.S == pytest.approx(S_AFTER_DAY10)
    assert state.last_review == 10


def test_stage3_studyforge_06_exponential_matches_topic_state_update():
    ts = TopicState(name="t", difficulty=3)
    model = ExponentialModel()
    state = ModelState(S=ts.S, last_review=None, difficulty=3)
    for day, R in ((3, None), (10, None), (14, 0.8)):
        ts.review(day, R=R)
        state = model.transition(state, day, outcome=R)
        assert state.S == pytest.approx(ts.S)
        assert state.last_review == ts.last_review


def test_stage3_studyforge_06_exponential_fit_matches_fit_stability():
    model = ExponentialModel()
    # Four events yield three usable (interval, recall) pairs: the
    # first event anchors the intervals and has none of its own.
    events = _events([(2, 0.45), (4, 0.30), (6, 0.14), (8, 0.10)])
    fitted = model.fit(events, difficulty=3)
    assert isinstance(fitted, ModelState)
    assert fitted.last_review == 8
    # Same pairs through the existing fit: least squares on log recall.
    from studyforge.memory import fit_stability

    pairs = [(2.0, 0.30), (2.0, 0.14), (2.0, 0.10)]
    assert fitted.S == pytest.approx(fit_stability(pairs))
    # Too few usable pairs: the default stands, marked by last_review.
    sparse = model.fit(_events([(2, 0.5)]), difficulty=3)
    assert sparse.S == pytest.approx(1.0)
    assert sparse.last_review == 2


def test_stage3_studyforge_06_model_serialization_round_trips():
    model = ExponentialModel()
    state = model.fit(_events([(2, 0.45), (4, 0.30), (6, 0.14)]),
                      difficulty=3)
    data = model.to_dict(state)
    assert data["model"] == "exponential/1.0"
    restored_model, restored = load_model(data)
    assert restored_model.version == model.version
    assert restored.S == pytest.approx(state.S)
    assert restored.last_review == state.last_review
    assert restored_model.predict(restored, 3.0) == pytest.approx(
        model.predict(state, 3.0))
    with pytest.raises(ValueError):
        load_model({"model": "no-such-model/9.9", "S": 1.0,
                    "last_review": None, "difficulty": 3})
    with pytest.raises(ValueError):
        load_model({"S": 1.0})


def test_stage3_studyforge_06_model_registry_resolves_names():
    assert isinstance(get_model("exponential"), ExponentialModel)
    assert isinstance(get_model("hlr"), HierarchicalHalfLifeModel)
    assert isinstance(get_model("fsrs"), FSRSAdapter)
    assert get_model("exponential").version == "exponential/1.0"
    with pytest.raises(ValueError):
        get_model("no-such-model")


# --- Outcome semantics and the chronology guard ---


def test_stage3_studyforge_06_outcome_normalization():
    assert normalize_outcome(1).kind == "binary"
    assert normalize_outcome(1).value == 1.0
    assert normalize_outcome(0).kind == "binary"
    assert normalize_outcome(0.5).kind == "fractional"
    assert normalize_outcome(0.5).value == 0.5
    assert normalize_outcome(None).kind == "missing"
    assert normalize_outcome(None).value is None
    for bad in (1.5, -0.1, float("nan"), float("inf"), True, "0.5"):
        with pytest.raises(ValueError):
            normalize_outcome(bad)


def test_stage3_studyforge_06_missing_outcomes_stay_unknown():
    model = ExponentialModel()
    # A logged review with no observed recall still moves last_review
    # forward, but contributes no (interval, recall) pair to the fit.
    events = _events([(2, None), (4, None), (6, None)])
    fitted = model.fit(events, difficulty=3)
    assert fitted.S == pytest.approx(1.0)  # default: no usable pairs
    assert fitted.last_review == 6
    state = ModelState(S=1.0, last_review=None, difficulty=3)
    state = model.transition(state, 2, outcome=None)
    # Unknown outcome at review: the model uses its own prediction
    # (first review, R = 0), exactly as TopicState.review does.
    assert state.S == pytest.approx(2.0)


def test_stage3_studyforge_06_chronology_guard_asof_excludes_future():
    model = ExponentialModel()
    early = _events([(2, 0.45), (4, 0.30), (6, 0.14)])
    late = early + [HistoryEvent(day=9, outcome=0.02)]
    fit_early = model.fit(early, difficulty=3)
    fit_asof = model.fit(late, as_of=7, difficulty=3)
    fit_all = model.fit(late, difficulty=3)
    # The day-9 event must not leak into the as-of-7 fit.
    assert fit_asof.S == pytest.approx(fit_early.S)
    assert fit_asof.last_review == 6
    assert fit_all.last_review == 9
    assert fit_all.S != pytest.approx(fit_early.S)


def test_stage3_studyforge_06_chronology_guard_transition_cannot_go_back():
    model = ExponentialModel()
    state = ModelState(S=2.0, last_review=10, difficulty=3)
    with pytest.raises(ValueError):
        model.transition(state, 9)
    # A same-day repeat review is a no-op, matching simulate.
    same = model.transition(state, 10)
    assert same.S == pytest.approx(2.0)
    assert same.last_review == 10
    with pytest.raises(ValueError):
        model.transition(state, -1)
    with pytest.raises(ValueError):
        model.fit([HistoryEvent(day=-1, outcome=0.5)], difficulty=3)


def test_stage3_studyforge_06_probability_sanity_rules():
    model = ExponentialModel()
    state = ModelState(S=1.0, last_review=3, difficulty=3)
    for elapsed in (0.0, 1.0, 100.0):
        p = model.predict(state, elapsed)
        assert math.isfinite(p) and 0.0 <= p <= 1.0
    # Negative elapsed is clamped to zero, matching memory.recall.
    assert model.predict(state, -2.0) == 1.0
    with pytest.raises(ValueError):
        ModelState(S=0.0, last_review=None, difficulty=3)
    with pytest.raises(ValueError):
        ModelState(S=float("nan"), last_review=None, difficulty=3)


# --- Hierarchical half-life regression ---


def test_stage3_studyforge_06_hlr_no_data_matches_exponential():
    hlr = HierarchicalHalfLifeModel()
    assert hlr.version == "hlr/1.0"
    exp = ExponentialModel()
    state_h = hlr.fit(_events([(2, None)]), difficulty=3)
    state_e = exp.fit(_events([(2, None)]), difficulty=3)
    assert state_h.S == pytest.approx(state_e.S)
    assert hlr.predict(state_h, 3.0) == pytest.approx(
        exp.predict(state_e, 3.0))


def test_stage3_studyforge_06_hlr_shrinks_toward_pooled_value():
    events_a, events_b = _hlr_pairs_events()
    pooled = HierarchicalHalfLifeModel.fit_pooled([events_a, events_b])
    # Pooled S is 8/3 (rate 0.375); the pooled half-life is S * ln 2.
    assert pooled.pooled_half_life == pytest.approx(8.0 / 3.0 * math.log(2))
    fitted = pooled.fit(events_a, difficulty=3)
    # Shrunk rate (3 * 0.5 + 4 * 0.375) / 7 = 3/7, so S = 7/3: between
    # the per-topic 2.0 and the pooled 8/3.
    assert fitted.S == pytest.approx(7.0 / 3.0)
    assert 2.0 < fitted.S < 8.0 / 3.0


def test_stage3_studyforge_06_hlr_sparse_topic_uses_pooled_value():
    events_a, events_b = _hlr_pairs_events()
    pooled = HierarchicalHalfLifeModel.fit_pooled([events_a, events_b])
    sparse = pooled.fit(_events([(2, 0.5)]), difficulty=3)
    # One usable pair is below the fit threshold: the pooled value
    # stands instead of the noisy per-topic estimate or the default.
    assert sparse.S == pytest.approx(8.0 / 3.0)


# --- FSRS-family reference adapter: units are never interchanged ---


def test_stage3_studyforge_06_fsrs_unit_conversion_is_exact():
    assert from_studyforge_stability(1.0) == pytest.approx(LN_10_OVER_9)
    assert to_studyforge_stability(LN_10_OVER_9) == pytest.approx(1.0)
    for s in (0.5, 2.0, 3.939605):
        assert to_studyforge_stability(
            from_studyforge_stability(s)) == pytest.approx(s)


def test_stage3_studyforge_06_fsrs_predicts_ninety_percent_at_stability():
    adapter = FSRSAdapter()
    assert adapter.version == "fsrs-adapter/1.0"
    s_fsrs = from_studyforge_stability(2.0)
    state = ModelState(S=s_fsrs, last_review=0, difficulty=3)
    assert adapter.predict(state, s_fsrs) == pytest.approx(0.9)
    # The FSRS power-law shape differs from the exponential at the
    # same StudyForge stability: it forgets faster early, slower late.
    exp = ExponentialModel()
    exp_state = ModelState(S=2.0, last_review=0, difficulty=3)
    assert adapter.predict(state, 0.2) < exp.predict(exp_state, 0.2)
    assert adapter.predict(state, 20.0) > exp.predict(exp_state, 20.0)
    # The constants are the published FSRS-4.5/5/6 retrievability form.
    assert FSRS_FACTOR == pytest.approx(19.0 / 81.0)
    assert FSRS_DECAY == pytest.approx(0.5)


def test_stage3_studyforge_06_fsrs_needs_item_level_binary_data():
    adapter = FSRSAdapter()
    # Topic-level fractional outcomes are not compatible: FSRS fits
    # per-item binary grades, so the adapter refuses instead of
    # silently fitting the wrong target.
    with pytest.raises(ValueError):
        adapter.fit(_events([(2, 0.45), (4, 0.30), (6, 0.14)]),
                    difficulty=3)
    with pytest.raises(ValueError):
        adapter.fit([HistoryEvent(day=2, outcome=1.0),
                     HistoryEvent(day=4, outcome=0.0),
                     HistoryEvent(day=6, outcome=1.0)], difficulty=3)
    # Item-level binary outcomes fit, and the fitted stability is
    # reported in FSRS units, not StudyForge units.
    item_events = [HistoryEvent(day=2, outcome=1.0, item="q1", binary=True),
                   HistoryEvent(day=4, outcome=0.0, item="q2", binary=True),
                   HistoryEvent(day=6, outcome=1.0, item="q3", binary=True)]
    fitted = adapter.fit(item_events, difficulty=3)
    assert fitted.S == pytest.approx(
        from_studyforge_stability(ExponentialModel().fit(
            _events([(2, 1.0), (4, 0.0), (6, 1.0)]),
            difficulty=3).S))
    with pytest.raises(ValueError):
        adapter.transition(ModelState(S=1.0, last_review=None, difficulty=3),
                           2, outcome=0.5)


# --- Held-out metrics: hand-checked values, ranges, calibration, CIs ---


def test_stage3_studyforge_06_metric_values_are_hand_checked():
    assert log_loss([0.5, 0.5], [1, 0]) == pytest.approx(math.log(2))
    assert brier_score([0.5, 0.5], [1, 0]) == pytest.approx(0.25)
    assert mae([0.5, 0.5], [1.0, 0.0]) == pytest.approx(0.5)
    assert mse([0.5, 0.5], [1.0, 0.0]) == pytest.approx(0.25)
    assert log_loss([1.0], [1]) == pytest.approx(0.0)
    assert brier_score([1.0], [1]) == pytest.approx(0.0)
    # Perfect confidence in the wrong answer is heavily penalized,
    # and metric inputs are validated, not silently clipped.
    assert log_loss([0.0], [1]) > 10.0
    with pytest.raises(ValueError):
        log_loss([1.5], [1])
    with pytest.raises(ValueError):
        brier_score([0.5], [2])
    with pytest.raises(ValueError):
        mae([0.5], [1.5])


def test_stage3_studyforge_06_confidence_intervals():
    mean, lo, hi = mean_ci([0.0, 1.0, 0.0, 1.0])
    assert mean == pytest.approx(0.5)
    assert lo < 0.5 < hi
    # The mean CI is unclipped by design: log loss is unbounded
    # above, so its interval must be allowed past 1.
    assert lo < 0.0
    lo, hi = wilson_ci(8, 10)
    assert 0.0 <= lo < 0.8 < hi <= 1.0
    lo, hi = wilson_ci(0, 10)
    assert lo == 0.0 and hi < 0.5


def test_stage3_studyforge_06_calibration_bins_cover_all_predictions():
    preds = [0.05 + 0.1 * i for i in range(10)]
    outcomes = [0, 0, 0, 0, 1, 0, 1, 1, 1, 1]
    bins = calibration_bins(preds, outcomes, n_bins=5)
    assert sum(b["n"] for b in bins) == 10
    assert all(b["n"] == 2 for b in bins)
    for i, b in enumerate(bins):
        lo, hi = b["ci_lo"], b["ci_hi"]
        assert 0.0 <= lo <= b["observed"] <= hi <= 1.0
        assert b["predicted"] == pytest.approx(
            sum(preds[2 * i:2 * i + 2]) / 2)
    # The best-calibrated bin is the one whose predicted mean matches
    # its observed rate most closely; the worst is the furthest.
    assert bins[0]["observed"] == pytest.approx(0.0)
    assert bins[4]["observed"] == pytest.approx(1.0)


# --- Tournament: deterministic, version-tagged, honest about exclusions ---


def _tournament_topics():
    frac = [
        HistoryEvent(day=1, outcome=0.9),
        HistoryEvent(day=3, outcome=0.7),
        HistoryEvent(day=5, outcome=0.5),
        HistoryEvent(day=8, outcome=0.4),
        HistoryEvent(day=12, outcome=0.3),
        HistoryEvent(day=16, outcome=0.2),
    ]
    binary = [
        HistoryEvent(day=1, outcome=1.0, item="q1", binary=True),
        HistoryEvent(day=3, outcome=1.0, item="q2", binary=True),
        HistoryEvent(day=5, outcome=0.0, item="q3", binary=True),
        HistoryEvent(day=8, outcome=1.0, item="q4", binary=True),
        HistoryEvent(day=12, outcome=0.0, item="q5", binary=True),
        HistoryEvent(day=16, outcome=1.0, item="q6", binary=True),
    ]
    return [("alpha", 3, frac), ("beta", 2, binary)]


def test_stage3_studyforge_06_tournament_scores_all_models():
    models = [get_model("exponential"), get_model("hlr"),
              get_model("fsrs")]
    result = tournament(models, _tournament_topics(), cutoff=6)
    assert result["cutoff"] == 6
    versions = [r["model"] for r in result["results"]]
    assert versions == ["exponential/1.0", "hlr/1.0", "fsrs-adapter/1.0"]
    for r in result["results"]:
        assert r["n_binary"] == 3  # beta's day 8, 12, 16 events
        assert r["log_loss_ci"][0] <= r["log_loss"] <= r["log_loss_ci"][1]
        assert sum(b["n"] for b in r["calibration"]) == (
            r["n_binary"] + r["n_fractional"])
        for p in r["predictions"]:
            assert p["model"] == r["model"]
            assert 0.0 <= p["predicted"] <= 1.0
            assert p["day"] >= 6  # held out only: no train event scored
    by_version = {r["model"]: r for r in result["results"]}
    # The exponential and HLR models score both topics.
    for v in ("exponential/1.0", "hlr/1.0"):
        r = by_version[v]
        assert r["n_fractional"] == 3  # alpha's day 8, 12, 16 events
        assert math.isfinite(r["log_loss"]) and 0.0 <= r["brier"] <= 1.0
        assert math.isfinite(r["mae"]) and 0.0 <= r["mae"] <= 1.0
        assert math.isfinite(r["mse"]) and 0.0 <= r["mse"] <= 1.0
        assert r["mae_ci"][0] <= r["mae"] <= r["mae_ci"][1]
        assert r["topics"] == ["alpha", "beta"]
        assert r["skipped"] == []
    # FSRS scores only the item-level binary topic; the fractional
    # topic is skipped for it, with the reason recorded.
    fsrs = by_version["fsrs-adapter/1.0"]
    assert fsrs["n_fractional"] == 0
    assert fsrs["topics"] == ["beta"]
    assert len(fsrs["skipped"]) == 1
    assert fsrs["skipped"][0]["topic"] == "alpha"
    assert "item-level" in fsrs["skipped"][0]["reason"]
    assert result["excluded"] == []


def test_stage3_studyforge_06_tournament_is_deterministic():
    models = [get_model("exponential"), get_model("hlr")]
    first = tournament(models, _tournament_topics(), cutoff=6)
    second = tournament(models, _tournament_topics(), cutoff=6)
    assert first == second


def test_stage3_studyforge_06_tournament_excludes_fsrs_without_item_data():
    frac_only = [("alpha", 3, _tournament_topics()[0][2])]
    result = tournament([get_model("exponential"), get_model("fsrs")],
                        frac_only, cutoff=6)
    assert [r["model"] for r in result["results"]] == ["exponential/1.0"]
    assert len(result["excluded"]) == 1
    assert result["excluded"][0]["model"] == "fsrs-adapter/1.0"
    assert "item-level" in result["excluded"][0]["reason"]


def test_stage3_studyforge_06_tournament_never_scores_train_events():
    models = [get_model("exponential")]
    topics = _tournament_topics()
    result = tournament(models, topics, cutoff=6)
    scored_days = sorted(p["day"] for p in result["results"][0]["predictions"])
    assert scored_days == [8, 8, 12, 12, 16, 16]
    # Moving a held-out outcome cannot change the fit: the fit sees
    # only pre-cutoff events.
    moved = [("alpha", 3, [HistoryEvent(day=1, outcome=0.9),
                           HistoryEvent(day=3, outcome=0.7),
                           HistoryEvent(day=5, outcome=0.5),
                           HistoryEvent(day=8, outcome=0.99),
                           HistoryEvent(day=12, outcome=0.01),
                           HistoryEvent(day=16, outcome=0.2)])]
    r2 = tournament(models, moved, cutoff=6)
    assert r2["results"][0]["predictions"][0]["predicted"] == pytest.approx(
        result["results"][0]["predictions"][0]["predicted"])


# --- Backward compatibility: the planner and dashboard are unchanged by default ---


def test_stage3_studyforge_06_planner_default_is_unchanged():
    topic = PlanTopic("probability", [("Midterm", 0.6, 7),
                                     ("Final", 0.5, 21)],
                      S=S_AFTER_DAY10, last_review=10, difficulty=3,
                      history=[(3, None), (10, None)])
    default_s, default_r = simulate(topic, [12, 15])
    model_s, model_r = simulate(topic, [12, 15],
                                model=ExponentialModel())
    assert model_s == pytest.approx(default_s)
    assert model_r == default_r
    # The as-of day-7 recall still comes from the day-3 state only.
    _, (r_mid, r_fin) = simulate(topic, [])
    assert r_mid == pytest.approx(RECALL_DAY7)
    assert r_fin == pytest.approx(math.exp(-(21 - 10) / S_AFTER_DAY10))


def test_stage3_studyforge_06_example_plan_is_unchanged():
    cfg = st.load_config("exams.yaml")
    topics = dash.fresh_topics(cfg)
    horizon = max(t.latest_exam_day for t in topics)
    plan, obj = build_plan(topics, horizon, cfg["daily_budget_minutes"],
                           cfg["session_minutes"], set(cfg["rest_days"]),
                           start_day=0, seed=0)
    assert sum(len(v) for v in plan.values()) == 72
    # A05 moved the example objective from 2.278051 to 2.312646: the
    # exchange phase finds swaps single moves cannot see. The point
    # of this test stands: the default and HLR plans still agree.
    assert obj == pytest.approx(2.312646)
    plan2, obj2 = build_plan(topics, horizon, cfg["daily_budget_minutes"],
                             cfg["session_minutes"], set(cfg["rest_days"]),
                             start_day=0, seed=0, model=get_model("hlr"))
    assert sum(len(v) for v in plan2.values()) == 72
    assert obj2 == pytest.approx(2.312646)


def test_stage3_studyforge_06_dashboard_trajectory_uses_model():
    default = dash.recall_trajectory("rotation", [3, 7], 1.0, 14, 14)
    modeled = dash.recall_trajectory("rotation", [3, 7], 1.0, 14, 14,
                                     model=ExponentialModel())
    assert modeled == default
    fsrs = dash.recall_trajectory("rotation", [3, 7], 1.0, 14, 14,
                                  model=FSRSAdapter())
    assert fsrs["S_final"] != pytest.approx(default["S_final"])
    assert all(0.0 <= y <= 1.0 for y in fsrs["ys"])


def test_stage3_studyforge_06_cli_model_flag_tags_state(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    exams_path = str(__import__("pathlib").Path(__file__).resolve()
                     .parent.parent / "exams.yaml")
    state_path = tmp_path / "state.json"
    _cli.main(["--state", str(state_path), "plan", "--exams",
               exams_path, "--model", "hlr"])
    saved = json.loads(state_path.read_text())
    assert saved["model_version"] == "hlr/1.0"
    _cli.main(["--state", str(state_path), "plan", "--exams",
               exams_path])
    saved = json.loads(state_path.read_text())
    assert saved["model_version"] == "exponential/1.0"
    with pytest.raises(SystemExit):
        _cli.main(["--state", str(state_path), "plan", "--exams",
                   exams_path, "--model", "no-such-model"])


def test_stage3_studyforge_06_cli_plan_topics_carry_model():
    state = st.new_state({
        "exams": [{"name": "Midterm", "day": 7, "topics": [
            {"name": "Probability", "difficulty": 3, "weight": 0.6}]}],
        "rest_days": [], "daily_budget_minutes": 120,
        "session_minutes": 25,
    })
    topics = _cli._plan_topics(state, model=get_model("hlr"))
    assert topics[0].model is not None
    assert topics[0].model.version == "hlr/1.0"
    default_topics = _cli._plan_topics(state)
    assert default_topics[0].model is None


def test_stage3_studyforge_06_stored_version_tag_rebuilds(tmp_path,
                                                          monkeypatch):
    """A state tagged with a version tag rebuilds without a --model flag.

    States store the tag (``exponential/1.0``) while the CLI takes the
    registry name (``exponential``); the rebuild must bridge the two
    instead of treating the tag as a name.
    """
    from studyforge.models import get_model_by_version

    assert get_model_by_version("exponential/1.0").version == (
        "exponential/1.0")
    assert get_model_by_version("hlr/1.0").version == "hlr/1.0"
    assert get_model_by_version("fsrs-adapter/1.0").version == (
        "fsrs-adapter/1.0")
    with pytest.raises(ValueError):
        get_model_by_version("exponential/9.9")
    monkeypatch.chdir(tmp_path)
    exams_path = str(__import__("pathlib").Path(__file__).resolve()
                     .parent.parent / "exams.yaml")
    state_path = tmp_path / "state.json"
    _cli.main(["--state", str(state_path), "plan", "--exams", exams_path])
    saved = json.loads(state_path.read_text())
    assert saved["model_version"] == "exponential/1.0"
    # Replan and what-if with no --model flag must reuse the stored tag.
    _cli.main(["--state", str(state_path), "replan", "--today", "2"])
    _cli.main(["--state", str(state_path), "what-if", "--day", "1",
               "--extra-minutes", "60", "--today", "0"])
    saved = json.loads(state_path.read_text())
    assert saved["model_version"] == "exponential/1.0"
