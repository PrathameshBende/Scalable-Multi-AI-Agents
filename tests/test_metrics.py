import math

import pytest

from scalable_agents.metrics import (
    evaluate_run, hit_at_k, mcnemar_exact, mrr_at_k, ndcg_at_k, recall_at_k, wilson_interval,
)


def test_hit_recall_mrr():
    ranked = ["d3", "d1", "d7", "d2"]
    assert hit_at_k(ranked, {"d1"}, 1) == 0.0
    assert hit_at_k(ranked, {"d1"}, 2) == 1.0
    assert recall_at_k(ranked, {"d1", "d2"}, 2) == 0.5
    assert recall_at_k(ranked, {"d1", "d2"}, 4) == 1.0
    assert mrr_at_k(ranked, {"d7"}, 10) == pytest.approx(1 / 3)
    assert mrr_at_k(ranked, {"d9"}, 10) == 0.0


def test_ndcg_hand_computed():
    # relevant d1 at rank 2: DCG = 1/log2(3); ideal = 1/log2(2) = 1
    assert ndcg_at_k(["d3", "d1"], {"d1": 1}, 10) == pytest.approx(1 / math.log2(3))
    assert ndcg_at_k(["d1", "d3"], {"d1": 1}, 10) == pytest.approx(1.0)
    assert ndcg_at_k(["x"], {}, 10) == 0.0


def test_evaluate_run_counts_missing_queries_as_zero():
    qrels = {"q1": {"a": 1}, "q2": {"b": 1}}
    m = evaluate_run({"q1": ["a"]}, qrels, ks=(1,))
    assert m["hit@1"] == 0.5
    assert m["n_queries"] == 2


def test_wilson_interval_contains_point_estimate():
    lo, hi = wilson_interval(27, 30)
    assert lo < 0.9 < hi
    assert 0.73 < lo < 0.75 and 0.96 < hi < 0.98  # known values for 27/30
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_mcnemar():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 5) == 1.0
    assert mcnemar_exact(10, 0) == pytest.approx(2 / 2**10)
    assert mcnemar_exact(0, 10) == mcnemar_exact(10, 0)
