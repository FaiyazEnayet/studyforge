import math

import pytest

from studyforge.memory import GROWTH, TopicState, initial_stability, recall


def test_recall_at_zero_is_one():
    assert recall(0, 2.0) == pytest.approx(1.0)


def test_recall_decays_monotonically():
    rs = [recall(t, 2.0) for t in [0, 1, 2, 5, 10]]
    assert all(a > b for a, b in zip(rs, rs[1:]))


def test_recall_matches_formula():
    assert recall(3.0, 2.0) == pytest.approx(math.exp(-1.5))


def test_higher_stability_decays_slower():
    assert recall(5.0, 10.0) > recall(5.0, 1.0)


def test_recall_never_negative_and_handles_zero_stability():
    assert recall(5.0, 0.0) == 0.0
    assert recall(-3.0, 2.0) == pytest.approx(1.0)


def test_initial_stability_decreases_with_difficulty():
    ss = [initial_stability(d) for d in range(1, 6)]
    assert all(a > b for a, b in zip(ss, ss[1:]))
    assert initial_stability(1) == pytest.approx(3.0)
    assert initial_stability(5) == pytest.approx(0.6)


def test_review_increases_stability():
    ts = TopicState("x", difficulty=3)
    s0 = ts.S
    ts.review(5)
    assert ts.S > s0
    assert ts.last_review == 5


def test_review_gain_grows_with_decay():
    early = TopicState("x", difficulty=3)
    late = TopicState("x", difficulty=3)
    early.review(0)  # first exposure for both
    late.review(0)
    early.review(1)   # high retrievability
    late.review(10)   # low retrievability
    assert late.S > early.S


def test_fresh_topic_has_zero_recall():
    ts = TopicState("x", difficulty=3)
    assert ts.recall_on(5) == 0.0


def test_first_review_gets_full_gain():
    ts = TopicState("x", difficulty=3)
    s0 = ts.S
    R = ts.review(4)
    assert R == 0.0
    assert ts.S == pytest.approx(s0 * (1 + GROWTH))


def test_full_decay_doubles_stability():
    ts = TopicState("x", difficulty=3)
    s0 = ts.S
    ts.review(10_000)  # R ~ 0
    assert ts.S == pytest.approx(s0 * (1 + GROWTH))


def test_same_day_review_gives_no_gain():
    ts = TopicState("x", difficulty=3)
    ts.review(0)  # first exposure: full gain
    s1 = ts.S
    ts.review(0)  # immediate repeat: nothing new proven
    assert ts.S == pytest.approx(s1)


def test_review_returns_retrievability():
    ts = TopicState("x", difficulty=3)
    ts.review(0)  # first exposure
    R = ts.review(2)
    assert 0.0 < R < 1.0
    assert R == pytest.approx(math.exp(-2 / (2 * initial_stability(3))))


def test_state_roundtrip():
    ts = TopicState("x", difficulty=4)
    ts.review(3)
    d = ts.to_dict()
    ts2 = TopicState.from_dict(d)
    assert ts2.name == "x"
    assert ts2.S == pytest.approx(ts.S)
    assert ts2.last_review == 3
