"""Retrieval strategies with a fake bag-of-words embedder and a scripted fake LLM."""

import numpy as np
import pytest

from scalable_agents.retrieval import (
    HYDE_PROMPT, BM25Index, DenseIndex, Retriever, tokenize,
)

DOCS = {
    "d1": "vitamin d supplements improve bone density in older adults",
    "d2": "statins lower cholesterol and reduce heart attack risk",
    "d3": "deep sleep consolidates memory in the hippocampus",
    "d4": "calcium and vitamin d reduce fracture risk",
    "d5": "aspirin thins the blood",
}
VOCAB = sorted({w for t in DOCS.values() for w in tokenize(t)} | {"osteoporosis", "skeleton"})


def fake_embed(texts):
    m = np.zeros((len(texts), len(VOCAB)), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in tokenize(t):
            if w in VOCAB:
                m[i, VOCAB.index(w)] += 1
    return m


@pytest.fixture
def indexes():
    ids, texts = list(DOCS), list(DOCS.values())
    return DenseIndex(fake_embed).build(ids, texts), BM25Index().build(ids, texts)


def test_dense_finds_lexically_closest(indexes):
    dense, _ = indexes
    assert dense.search("statins cholesterol", 1) == ["d2"]
    assert len(dense.search("anything", 99)) == len(DOCS)  # k capped at corpus size


def test_bm25(indexes):
    _, bm25 = indexes
    assert bm25.search("hippocampus memory", 1) == ["d3"]


def test_hyde_uses_generated_passage(indexes):
    dense, bm25 = indexes
    prompts = []

    def gen(prompt):
        prompts.append(prompt)
        return "vitamin d and calcium strengthen bone density"

    r = Retriever(dense, bm25, gen)
    # The query has no word in common with any doc, so only the passage can help.
    ranked = r.search("osteoporosis skeleton", 2, "hyde")
    assert set(ranked) == {"d1", "d4"}
    assert prompts == [HYDE_PROMPT.format(query="osteoporosis skeleton")]
    assert r.last_trace["hyde_passages"] == ["vitamin d and calcium strengthen bone density"]


def test_rag_fusion_merges_rephrasings(indexes):
    dense, bm25 = indexes
    # fusion_depth=1: each query's top hit votes once (in a 5-doc corpus deeper ranks
    # would just be zero-score ties).
    r = Retriever(dense, bm25, lambda p: "1. statins cholesterol\n2. aspirin blood",
                  n_queries=2, fusion_depth=1)
    ranked = r.search("heart medicine", 3, "rag_fusion")
    assert r.last_trace["queries"] == ["heart medicine", "statins cholesterol", "aspirin blood"]
    assert ranked == ["d2", "d5"]  # d2 top for two queries, d5 for one


def test_hybrid_and_errors(indexes):
    dense, bm25 = indexes
    r = Retriever(dense, bm25)
    assert r.search("vitamin d fracture", 1, "hybrid") == ["d4"]
    with pytest.raises(ValueError):
        r.search("x", 1, "hyde")  # no LLM
    with pytest.raises(ValueError):
        r.search("x", 1, "nope")
    with pytest.raises(ValueError):
        Retriever(dense).search("x", 1, "hybrid")


def test_hyde_multiple_samples_passes_sample_index(indexes):
    dense, bm25 = indexes
    calls = []

    def gen(prompt, sample=0):
        calls.append(sample)
        return ["vitamin d bone density", "calcium fracture risk", "vitamin d calcium"][sample]

    r = Retriever(dense, bm25, gen, hyde_samples=3)
    assert set(r.search("osteoporosis skeleton", 2, "hyde")) == {"d1", "d4"}
    assert calls == [0, 1, 2] and len(r.last_trace["hyde_passages"]) == 3


def test_hybrid_variants_combine_llm_and_bm25(indexes):
    dense, bm25 = indexes
    r = Retriever(dense, bm25, lambda p: "statins cholesterol", n_queries=1)
    assert r.search("heart attack", 1, "hyde_hybrid") == ["d2"]
    assert r.search("heart attack", 1, "fusion_hybrid") == ["d2"]
    with pytest.raises(ValueError):
        Retriever(dense, None, lambda p: "x").search("q", 1, "hyde_hybrid")
