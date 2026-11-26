import pytest

from scalable_agents.fusion import parse_query_list, reciprocal_rank_fusion


def test_rrf_hand_computed():
    a = ["d1", "d2", "d3"]
    b = ["d3", "d1", "d4"]
    fused = reciprocal_rank_fusion([a, b], k=60)
    # d1: 1/61 + 1/62, d3: 1/63 + 1/61, d2: 1/62, d4: 1/63
    assert fused == ["d1", "d3", "d2", "d4"]


def test_rrf_doc_in_both_lists_beats_top_of_one():
    fused = reciprocal_rank_fusion([["x", "shared"], ["y", "shared"]], k=60)
    assert fused[0] == "shared"


def test_rrf_ignores_duplicates_within_a_list_and_truncates():
    assert reciprocal_rank_fusion([["a", "a", "b"]], top_n=1) == ["a"]
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"]], k=-1)


def test_rrf_is_deterministic_on_ties():
    assert reciprocal_rank_fusion([["b"], ["a"]]) == ["a", "b"]


def test_parse_query_list_strips_numbering_and_preamble():
    reply = "Here are the queries:\n1. vitamin D bone density\n- \"calcium intake fractures\"\n2) vitamin D bone density\n\n* elderly falls"
    assert parse_query_list(reply, 3) == [
        "vitamin D bone density", "calcium intake fractures", "elderly falls",
    ]
    assert parse_query_list(reply, 1) == ["vitamin D bone density"]
