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
| hyde_hybrid | HyDE ranking + bm25 ranking merged with RRF                     | 1         |
| fusion_hybrid | each rephrasing searched with dense AND bm25; all merged by RRF | 1       |

Any strategy can be followed by a cross-encoder reranking step: ``"<base>+rerank@N"``
takes the base strategy's top N and re-sorts them by a reranker score for
(original query, document text). A cross-encoder reads query and document together,
so it judges relevance better than comparing two separately computed embeddings,
but it costs one model call per (query, doc) pair, so it only sees the top N.
"""

from __future__ import annotations

import re
from typing import Callable, Sequence

import numpy as np

from .fusion import parse_query_list, reciprocal_rank_fusion

Embedder = Callable[[list[str]], np.ndarray]
Generator = Callable[[str], str]
# (query, [doc texts]) -> one relevance score per doc, higher = more relevant.
Reranker = Callable[[str, list[str]], Sequence[float]]

# HyDE prompt for SciFact, taken from the HyDE paper's released prompts
# (Gao et al., 2022, "Precise Zero-Shot Dense Retrieval without Relevance Labels").
HYDE_PROMPT = (
    "Please write a scientific paper passage to support/refute the claim.\n"
    "Claim: {query}\nPassage:"
)

# For question-style datasets (e.g. NFCorpus), following the HyDE paper's prompts for
# question datasets such as TREC-COVID. Fixed before any NFCorpus run.
HYDE_PROMPT_QUESTION = (
    "Please write a scientific paper passage to answer the question.\n"
    "Question: {query}\nPassage:"
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

    STRATEGIES = ("bm25", "dense", "hyde", "rag_fusion", "hybrid", "hyde_hybrid", "fusion_hybrid")

    def __init__(
        self,
        dense: DenseIndex,
        bm25: BM25Index | None = None,
        generate: Generator | None = None,
        n_queries: int = 3,
        fusion_depth: int = 100,
        rrf_k: int = 60,
        hyde_samples: int = 1,
        rerank: Reranker | None = None,
        doc_text: dict[str, str] | None = None,
        hyde_prompt: str = HYDE_PROMPT,
    ):
        self.dense = dense
        self.bm25 = bm25
        self.generate = generate
        self.n_queries = n_queries
        self.fusion_depth = fusion_depth
        self.rrf_k = rrf_k
        self.hyde_samples = hyde_samples
        self.rerank = rerank
        self.doc_text = doc_text
        self.hyde_prompt = hyde_prompt
        self.last_trace: dict = {}

    # --- individual strategies -------------------------------------------
    def _hyde_vector(self, query: str) -> np.ndarray:
        prompt = self.hyde_prompt.format(query=query)
        if self.hyde_samples == 1:
            passages = [self._gen(prompt)]
        else:  # extra samples are drawn with temperature > 0 (see CachedGenerator)
            passages = [self.generate(prompt, sample=i) for i in range(self.hyde_samples)]
        # As in the HyDE paper, the query's own embedding is averaged in with the
        # hypothetical documents, which guards against a bad generation.
        vecs = _normalize(self.dense.embed([query, *passages]))
        self.last_trace = {"hyde_passages": passages}
        return vecs.mean(axis=0)

    def hyde(self, query: str, k: int) -> list[str]:
        return self.dense.search_vector(self._hyde_vector(query), k)

    def hyde_hybrid(self, query: str, k: int) -> list[str]:
        """HyDE dense ranking fused (RRF) with BM25 on the original query."""
        if self.bm25 is None:
            raise ValueError("hyde_hybrid needs a BM25 index")
        rankings = [self.dense.search_vector(self._hyde_vector(query), self.fusion_depth),
                    self.bm25.search(query, self.fusion_depth)]
        return reciprocal_rank_fusion(rankings, k=self.rrf_k, top_n=k)

    def _multi_queries(self, query: str) -> list[str]:
        reply = self._gen(MULTI_QUERY_PROMPT.format(query=query, n=self.n_queries))
        queries = [query] + parse_query_list(reply, self.n_queries)
        self.last_trace = {"queries": queries}
        return queries

    def rag_fusion(self, query: str, k: int) -> list[str]:
        rankings = [self.dense.search(q, self.fusion_depth) for q in self._multi_queries(query)]
        return reciprocal_rank_fusion(rankings, k=self.rrf_k, top_n=k)

    def fusion_hybrid(self, query: str, k: int) -> list[str]:
        """RAG-Fusion where every rephrasing is searched with both dense and BM25."""
        if self.bm25 is None:
            raise ValueError("fusion_hybrid needs a BM25 index")
        rankings = []
        for q in self._multi_queries(query):
            rankings += [self.dense.search(q, self.fusion_depth), self.bm25.search(q, self.fusion_depth)]
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

    def reranked(self, query: str, k: int, base: str, depth: int) -> list[str]:
        """Top ``depth`` of ``base``, re-sorted by the reranker; ties keep the base order."""
        if self.rerank is None or self.doc_text is None:
            raise ValueError("reranking needs a reranker and doc_text")
        candidates = self.search(query, depth, base)
        trace = self.last_trace
        scores = list(self.rerank(query, [self.doc_text[d] for d in candidates]))
        if len(scores) != len(candidates):
            raise ValueError("reranker returned the wrong number of scores")
        order = sorted(range(len(candidates)), key=lambda i: (-scores[i], i))
        self.last_trace = {**trace, "rerank_depth": depth}
        return [candidates[i] for i in order[:k]]

    def search(self, query: str, k: int = 10, strategy: str = "dense") -> list[str]:
        self.last_trace = {}
        if "+rerank@" in strategy:
            base, depth = strategy.split("+rerank@")
            return self.reranked(query, k, base, int(depth))
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
        if strategy == "hyde_hybrid":
            return self.hyde_hybrid(query, k)
        if strategy == "fusion_hybrid":
            return self.fusion_hybrid(query, k)
        raise ValueError(f"unknown strategy {strategy!r}; choose from {self.STRATEGIES}")

    def _gen(self, prompt: str) -> str:
        if self.generate is None:
            raise ValueError("this strategy needs an LLM generator")
        return self.generate(prompt)
