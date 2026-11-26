"""Retrieval metrics.

All functions take a ranked list of document ids (best first) and the set or dict
of relevant ids for one query. ``evaluate_run`` averages them over many queries.

Definitions (k = cut-off):
- hit@k     : 1 if at least one relevant doc is in the top k, else 0.
- recall@k  : fraction of the relevant docs that appear in the top k.
- mrr@k     : 1 / rank of the first relevant doc in the top k (0 if none).
- ndcg@k    : discounted cumulative gain with linear gain (same as pytrec_eval,
              which BEIR uses), normalised by the ideal ranking.
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence


def hit_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    rel = set(relevant)
    return float(any(d in rel for d in ranked[:k]))


def recall_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    rel = set(relevant)
    if not rel:
        return 0.0
    return len(set(ranked[:k]) & rel) / len(rel)


def mrr_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    rel = set(relevant)
    for i, d in enumerate(ranked[:k], start=1):
        if d in rel:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], graded: Mapping[str, int], k: int) -> float:
    dcg = sum(
        graded.get(d, 0) / math.log2(i + 1) for i, d in enumerate(ranked[:k], start=1)
    )
    ideal = sorted((g for g in graded.values() if g > 0), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 1) for i, g in enumerate(ideal, start=1))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_run(
    run: Mapping[str, Sequence[str]],
    qrels: Mapping[str, Mapping[str, int]],
    ks: Sequence[int] = (1, 3, 5, 10),
) -> dict[str, float]:
    """Average metrics over every query that has relevance judgements.

    A query in ``qrels`` that is missing from ``run`` counts as an empty ranking
    (all zeros), so a strategy cannot look better by skipping hard queries.
    """
    qids = [q for q, rels in qrels.items() if any(g > 0 for g in rels.values())]
    if not qids:
        raise ValueError("qrels has no relevant documents")
    totals: dict[str, float] = {}
    for qid in qids:
        ranked = list(run.get(qid, []))
        graded = qrels[qid]
        relevant = [d for d, g in graded.items() if g > 0]
        for k in ks:
            for name, value in (
                (f"hit@{k}", hit_at_k(ranked, relevant, k)),
                (f"recall@{k}", recall_at_k(ranked, relevant, k)),
                (f"mrr@{k}", mrr_at_k(ranked, relevant, k)),
                (f"ndcg@{k}", ndcg_at_k(ranked, graded, k)),
            ):
                totals[name] = totals.get(name, 0.0) + value
    out = {name: v / len(qids) for name, v in totals.items()}
    out["n_queries"] = float(len(qids))
    return out


def per_query_hits(
    run: Mapping[str, Sequence[str]],
    qrels: Mapping[str, Mapping[str, int]],
    k: int,
) -> dict[str, float]:
    """hit@k for each query, used for paired significance tests."""
    return {
        q: hit_at_k(list(run.get(q, [])), [d for d, g in r.items() if g > 0], k)
        for q, r in qrels.items()
        if any(g > 0 for g in r.values())
    }


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion (better than +-1.96*SE for small n)."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_exact(a_only: int, b_only: int) -> float:
    """Two-sided exact McNemar p-value from the discordant pair counts.

    ``a_only`` = queries strategy A got right and B got wrong; ``b_only`` the reverse.
    Under the null both are equally likely, so this is a binomial test with p=0.5.
    """
    n = a_only + b_only
    if n == 0:
        return 1.0
    k = min(a_only, b_only)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2**n
    return min(1.0, 2 * tail)
