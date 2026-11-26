"""Indexes and retrieval strategies.

An *embedder* is any callable ``list[str] -> np.ndarray`` of shape (n, dim).
A *generator* is any callable ``str -> str`` (an LLM prompt -> completion).
Keeping both as plain callables lets the tests use fakes instead of real models.

Strategies compared in ``scripts/run_retrieval_eval.py``:

| name        | what it does                                                    | LLM calls |
|-------------|-----------------------------------------------------------------|-----------|
| bm25        | keyword search (reference point)                                | 0         |
| dense       | embed the query, cosine top-k  (the BASELINE)                   | 0         |
| hyde        | LLM writes a fake answer passage; search with its embedding     | 1         |
| rag_fusion  | LLM writes n rephrasings; search each; merge with RRF           | 1         |
| hybrid      | dense + bm25 merged with RRF                                    | 0         |
"""

from __future__ import annotations

import re
from typing import Callable, Sequence

import numpy as np

from .fusion import parse_query_list, reciprocal_rank_fusion

Embedder = Callable[[list[str]], np.ndarray]
Generator = Callable[[str], str]

# HyDE prompt for SciFact, taken from the HyDE paper's released prompts
# (Gao et al., 2022, "Precise Zero-Shot Dense Retrieval without Relevance Labels").
HYDE_PROMPT = (
    "Please write a scientific paper passage to support/refute the claim.\n"
    "Claim: {query}\nPassage:"
)

MULTI_QUERY_PROMPT = (
    "You help search a database of scientific paper abstracts.\n"
    "Write {n} different search queries that would find abstracts relevant to the "
    "claim below. Use different wording and focus on different aspects. "
    "Output one query per line with no numbering and no other text.\n"
    "Claim: {query}"
)


def _normalize(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    if m.ndim == 1:
        m = m[None, :]
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


class DenseIndex:
    """Exact cosine-similarity search over normalised vectors (brute force).

    Brute force is exact and fast enough for ~10^5 docs on one machine; at larger
    scale you would swap this for an ANN index (FAISS HNSW/IVF, or a vector DB).
    """

    def __init__(self, embed: Embedder):
        self.embed = embed
        self.ids: list[str] = []
        self.matrix = np.zeros((0, 0), dtype=np.float32)

    def build(self, ids: Sequence[str], texts: Sequence[str]) -> "DenseIndex":
        if len(ids) != len(texts):
            raise ValueError("ids and texts differ in length")
        self.ids = list(ids)
        self.matrix = _normalize(self.embed(list(texts)))
        return self

    def search_vector(self, vec: np.ndarray, k: int) -> list[str]:
        scores = self.matrix @ _normalize(vec)[0]
        k = min(k, len(self.ids))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.lexsort((top, -scores[top]))]  # by score, then index for ties
        return [self.ids[i] for i in top]

    def search(self, query: str, k: int) -> list[str]:
        return self.search_vector(self.embed([query])[0], k)


_TOKEN = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25Index:
    def __init__(self) -> None:
        self.ids: list[str] = []
        self._bm25 = None

    def build(self, ids: Sequence[str], texts: Sequence[str]) -> "BM25Index":
        from rank_bm25 import BM25Okapi

        self.ids = list(ids)
        self._bm25 = BM25Okapi([tokenize(t) for t in texts])
        return self

    def search(self, query: str, k: int) -> list[str]:
        scores = np.asarray(self._bm25.get_scores(tokenize(query)))
        k = min(k, len(self.ids))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.lexsort((top, -scores[top]))]
        return [self.ids[i] for i in top]


class Retriever:
    """Runs one of the strategies listed in the module docstring."""

    STRATEGIES = ("bm25", "dense", "hyde", "rag_fusion", "hybrid")

    def __init__(
        self,
        dense: DenseIndex,
        bm25: BM25Index | None = None,
        generate: Generator | None = None,
        n_queries: int = 3,
        fusion_depth: int = 100,
        rrf_k: int = 60,
    ):
        self.dense = dense
        self.bm25 = bm25
        self.generate = generate
        self.n_queries = n_queries
        self.fusion_depth = fusion_depth
        self.rrf_k = rrf_k
        self.last_trace: dict = {}

    # --- individual strategies -------------------------------------------
    def hyde(self, query: str, k: int) -> list[str]:
        passage = self._gen(HYDE_PROMPT.format(query=query))
        # As in the HyDE paper, the query's own embedding is averaged in with the
        # hypothetical document, which guards against a bad generation.
        vecs = _normalize(self.dense.embed([query, passage]))
        self.last_trace = {"hyde_passage": passage}
        return self.dense.search_vector(vecs.mean(axis=0), k)

    def rag_fusion(self, query: str, k: int) -> list[str]:
        reply = self._gen(MULTI_QUERY_PROMPT.format(query=query, n=self.n_queries))
        queries = [query] + parse_query_list(reply, self.n_queries)
        rankings = [self.dense.search(q, self.fusion_depth) for q in queries]
        self.last_trace = {"queries": queries}
        return reciprocal_rank_fusion(rankings, k=self.rrf_k, top_n=k)

    def hybrid(self, query: str, k: int) -> list[str]:
        if self.bm25 is None:
            raise ValueError("hybrid needs a BM25 index")
        rankings = [
            self.dense.search(query, self.fusion_depth),
            self.bm25.search(query, self.fusion_depth),
        ]
        self.last_trace = {}
        return reciprocal_rank_fusion(rankings, k=self.rrf_k, top_n=k)

    def search(self, query: str, k: int = 10, strategy: str = "dense") -> list[str]:
        self.last_trace = {}
        if strategy == "dense":
            return self.dense.search(query, k)
        if strategy == "bm25":
            if self.bm25 is None:
                raise ValueError("bm25 strategy needs a BM25 index")
            return self.bm25.search(query, k)
        if strategy == "hyde":
            return self.hyde(query, k)
        if strategy == "rag_fusion":
            return self.rag_fusion(query, k)
        if strategy == "hybrid":
            return self.hybrid(query, k)
        raise ValueError(f"unknown strategy {strategy!r}; choose from {self.STRATEGIES}")

    def _gen(self, prompt: str) -> str:
        if self.generate is None:
            raise ValueError("this strategy needs an LLM generator")
        return self.generate(prompt)
