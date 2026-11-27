"""Pick the reranking config by a rule fixed before any reranking run.

Inputs: train runs (SciFact train, first 200 queries), one per reranker model, each with
the 4 base strategies x depths {20, 50, 100}. Rule: highest hit@5 (the pre-declared
primary metric); ties -> higher nDCG@10; then smaller depth; then the smaller reranker
(MiniLM before bge); then the cheaper base (dense, hybrid, hyde_hybrid, fusion_hybrid). Prints a table and a final line 'CHOICE <base> <depth> <reranker>'.
"""

from __future__ import annotations

import glob
import json
import os

BASE_COST = {"dense": 0, "hybrid": 1, "hyde_hybrid": 2, "fusion_hybrid": 3}
RERANKER_COST = {"cross-encoder/ms-marco-MiniLM-L-6-v2": 0, "BAAI/bge-reranker-base": 1}


def main() -> None:
    rows = []
    for path in sorted(glob.glob("results/tuning/retrieval_train_rerank_*.json")):
        d = json.load(open(path, encoding="utf-8"))
        for strat, m in d["metrics"].items():
            if "+rerank@" not in strat:
                if d["reranker"] == "cross-encoder/ms-marco-MiniLM-L-6-v2":  # print baselines once
                    print(f"  baseline {strat:<14} hit@5 {m['hit@5']:.3f}  nDCG@10 {m['ndcg@10']:.3f}")
                continue
            base, depth = strat.split("+rerank@")
            rows.append({"base": base, "depth": int(depth), "reranker": d["reranker"],
                         "hit@5": m["hit@5"], "ndcg@10": m["ndcg@10"],
                         "s_per_query": d["seconds_per_query"][strat]})
    best = max(rows, key=lambda r: (round(r["hit@5"], 6), round(r["ndcg@10"], 6), -r["depth"],
                                    -RERANKER_COST.get(r["reranker"], 9), -BASE_COST.get(r["base"], 9)))
    for r in sorted(rows, key=lambda r: (r["reranker"], r["base"], r["depth"])):
        mark = "  <- chosen" if r is best else ""
        print(f"{r['reranker'].split('/')[-1]:<24} {r['base']:<14} @{r['depth']:<4} "
              f"hit@5 {r['hit@5']:.3f}  nDCG@10 {r['ndcg@10']:.3f}  {r['s_per_query']:.2f}s/q{mark}")
    os.makedirs("results/tuning", exist_ok=True)
    with open("results/tuning/rerank_choice.json", "w", encoding="utf-8") as f:
        json.dump({"rule": __doc__.strip().splitlines()[2:6], "table": rows, "chosen": best}, f, indent=2)
    print(f"CHOICE {best['base']} {best['depth']} {best['reranker']}")


if __name__ == "__main__":
    main()
