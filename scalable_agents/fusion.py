"""Reciprocal Rank Fusion (RRF) and multi-query parsing for RAG-Fusion.

RRF (Cormack, Clarke and Buettcher, SIGIR 2009) merges several ranked lists:

    score(d) = sum over lists L containing d of  1 / (k + rank_L(d))

with k = 60 by default. It only uses ranks, not raw scores, so it can merge
lists whose scores are on different scales (e.g. BM25 and cosine similarity).
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Sequence


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]], k: int = 60, top_n: int | None = None
) -> list[str]:
    if k < 0:
        raise ValueError("k must be non-negative")
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        seen: set[str] = set()
        for rank, doc_id in enumerate(ranking, start=1):
            if doc_id in seen:  # a list should not vote twice for the same doc
                continue
            seen.add(doc_id)
            scores[doc_id] += 1.0 / (k + rank)
    # Ties broken by doc id so results are deterministic.
    fused = sorted(scores, key=lambda d: (-scores[d], d))
    return fused if top_n is None else fused[:top_n]


_PREFIX = re.compile(r"^\s*(?:[-*•]|\d+[.)]|q\d+[:.)])\s*", re.IGNORECASE)


def parse_query_list(text: str, n: int) -> list[str]:
    """Turn an LLM's 'one query per line' reply into a clean list of at most n queries.

    Strips bullets, numbering and surrounding quotes, drops empty lines,
    preamble lines ending with ':' and exact duplicates.
    """
    out: list[str] = []
    for line in text.splitlines():
        line = _PREFIX.sub("", line).strip().strip('"').strip("'").strip()
        if not line or line.endswith(":"):
            continue
        if line.lower() not in (q.lower() for q in out):
            out.append(line)
        if len(out) == n:
            break
    return out
