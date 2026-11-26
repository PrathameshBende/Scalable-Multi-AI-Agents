"""Compare retrieval strategies on BEIR SciFact and write results/retrieval.json.

Pre-declared before running (so the headline cannot be cherry-picked):
  baseline       = dense (all-MiniLM-L6-v2, cosine, top-k)
  primary metric = hit@5   (secondary: ndcg@10, the standard BEIR metric)
  "gain"         = reported both as absolute points and as relative %.
Significance: exact McNemar test on per-query hit@5 vs the baseline.

Usage (Colab, after Ollama is serving the model):
  python scripts/run_retrieval_eval.py --llm qwen2.5:3b
  python scripts/run_retrieval_eval.py --limit 20      # quick smoke test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scalable_agents.data import download_beir, load_beir  # noqa: E402
from scalable_agents.metrics import evaluate_run, mcnemar_exact, per_query_hits  # noqa: E402
from scalable_agents.retrieval import BM25Index, DenseIndex, Retriever  # noqa: E402

PRIMARY = "hit@5"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="scifact")
    ap.add_argument("--split", default="test",
                    help="'train' for tuning prompts/settings; 'test' only for the final report")
    ap.add_argument("--embedder",default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--llm", default="qwen2.5:3b")
    ap.add_argument("--strategies", default="bm25,dense,hyde,rag_fusion,hybrid")
    ap.add_argument("--n-queries", type=int, default=3, help="rephrasings for rag_fusion")
    ap.add_argument("--hyde-samples", type=int, default=1, help="hypothetical passages per query")
    ap.add_argument("--rrf-k", type=int, default=60)
    ap.add_argument("--fusion-depth", type=int, default=100)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0, help="only the first N queries (0 = all)")
    ap.add_argument("--workers", type=int, default=4, help="parallel LLM requests")
    ap.add_argument("--out", default=None,
                    help="default: results/retrieval.json (test) or results/tuning/retrieval_train_<tag>.json")
    ap.add_argument("--tag", default="run", help="name for a tuning run on the train split")
    args = ap.parse_args()

    from scalable_agents.models import CachedGenerator, environment_info, sentence_transformer_embedder

    data = load_beir(download_beir(args.dataset, "data"), split=args.split)
    qids = sorted(data.queries, key=lambda q: int(q) if q.isdigit() else q)
    if args.limit:
        qids = qids[: args.limit]
    qrels = {q: data.qrels[q] for q in qids}
    print(f"{len(data.corpus)} docs, {len(qids)} queries")

    embed = sentence_transformer_embedder(args.embedder)
    doc_ids = list(data.corpus)
    t0 = time.time()
    dense = DenseIndex(embed).build(doc_ids, [data.corpus[d] for d in doc_ids])
    index_seconds = time.time() - t0
    bm25 = BM25Index().build(doc_ids, [data.corpus[d] for d in doc_ids])

    strategies = args.strategies.split(",")
    llm_strategies = ("hyde", "rag_fusion", "hyde_hybrid", "fusion_hybrid")
    needs_llm = any(s in llm_strategies for s in strategies)
    gen = CachedGenerator(args.llm, f"results/cache/{args.llm.replace(':', '_')}.jsonl") if needs_llm else None

    runs: dict[str, dict[str, list[str]]] = {}
    timing: dict[str, float] = {}
    examples: dict[str, dict] = {}
    for strat in strategies:
        def one(q: str, strat=strat):
            # A fresh Retriever per query keeps last_trace thread-safe; indexes are shared.
            r = Retriever(dense, bm25, gen, n_queries=args.n_queries, fusion_depth=args.fusion_depth,
                          rrf_k=args.rrf_k, hyde_samples=args.hyde_samples)
            ranked = r.search(data.queries[q], args.k, strat)
            return q, ranked, r.last_trace

        t0 = time.time()
        workers = args.workers if strat in llm_strategies else 1
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(one, qids))
        timing[strat] = (time.time() - t0) / len(qids)
        runs[strat] = {q: ranked for q, ranked, _ in results}
        q0, _, trace0 = results[0]
        examples[strat] = {"query": data.queries[q0], "top3": runs[strat][q0][:3],
                           "relevant": sorted(qrels[q0]), **trace0}
        print(f"{strat:>13}: done ({timing[strat]:.2f}s/query)", flush=True)

    metrics = {s: evaluate_run(runs[s], qrels) for s in strategies}
    base = metrics.get("dense")
    comparison = {}
    if base:
        base_hits = per_query_hits(runs["dense"], qrels, 5)
        for s in strategies:
            if s == "dense":
                continue
            hits = per_query_hits(runs[s], qrels, 5)
            a_only = sum(1 for q in hits if hits[q] and not base_hits[q])
            b_only = sum(1 for q in hits if base_hits[q] and not hits[q])
            comparison[s] = {
                "hit@5_abs_points": 100 * (metrics[s][PRIMARY] - base[PRIMARY]),
                "hit@5_rel_pct": 100 * (metrics[s][PRIMARY] / base[PRIMARY] - 1) if base[PRIMARY] else None,
                "ndcg@10_abs_points": 100 * (metrics[s]["ndcg@10"] - base["ndcg@10"]),
                "ndcg@10_rel_pct": 100 * (metrics[s]["ndcg@10"] / base["ndcg@10"] - 1) if base["ndcg@10"] else None,
                "queries_won_vs_baseline": a_only,
                "queries_lost_vs_baseline": b_only,
                "mcnemar_p_hit@5": mcnemar_exact(a_only, b_only),
            }

    out = {
        "experiment": "retrieval",
        "dataset": args.dataset, "split": args.split, "n_docs": len(data.corpus), "n_queries": len(qids),
        "embedder": args.embedder, "llm": args.llm if needs_llm else None,
        "baseline": "dense", "primary_metric": PRIMARY, "k": args.k,
        "settings": {"n_queries": args.n_queries, "hyde_samples": args.hyde_samples,
                     "rrf_k": args.rrf_k, "fusion_depth": args.fusion_depth},
        "index_build_seconds": index_seconds,
        "metrics": metrics, "vs_baseline": comparison,
        "seconds_per_query": timing, "llm_calls_made": gen.calls if gen else 0,
        "examples": examples, "env": environment_info(),
    }
    if args.out is None:
        args.out = ("results/retrieval.json" if args.split == "test"
                    else f"results/tuning/retrieval_{args.split}_{args.tag}.json")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"\n{'strategy':>13} {'hit@1':>7} {'hit@5':>7} {'hit@10':>7} {'ndcg@10':>8} {'mrr@10':>7}")
    for s in strategies:
        m = metrics[s]
        print(f"{s:>13} {m['hit@1']:7.3f} {m['hit@5']:7.3f} {m['hit@10']:7.3f} {m['ndcg@10']:8.3f} {m['mrr@10']:7.3f}")
    for s, c in comparison.items():
        print(f"{s:>13} vs dense: hit@5 {c['hit@5_abs_points']:+.1f} pts ({c['hit@5_rel_pct']:+.1f}%), "
              f"won {c['queries_won_vs_baseline']} lost {c['queries_lost_vs_baseline']}, p={c['mcnemar_p_hit@5']:.3f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
