#!/usr/bin/env bash
# Cross-encoder reranking experiment (protocol in results/TUNING_LOG.md, written before running).
# 1. SciFact train (200 queries): 4 bases x depths {20,50,100} x 2 rerankers.
# 2. scripts/select_rerank.py picks one config by the fixed rule.
# 3. That config runs ONCE on SciFact test (second look at that set, disclosed) and ONCE on
#    NFCorpus test (never used before), each next to the dense baseline.
# Needs Ollama serving qwen2.5:3b (HyDE / RAG-Fusion generations).
set -x
cd "$(dirname "$0")/.."
BASES="dense hybrid hyde_hybrid fusion_hybrid"
STRATS=$(echo $BASES | tr ' ' ',')
for b in $BASES; do for d in 20 50 100; do STRATS="$STRATS,$b+rerank@$d"; done; done
for R in cross-encoder/ms-marco-MiniLM-L-6-v2 BAAI/bge-reranker-base; do
  python scripts/run_retrieval_eval.py --split train --limit 200 --strategies "$STRATS" \
    --reranker "$R" --tag "rerank_$(basename $R)"
done
python scripts/select_rerank.py > results/tuning/rerank_select.txt
cat results/tuning/rerank_select.txt
read BASE DEPTH RERANKER <<< "$(awk '/^CHOICE/ {print $2, $3, $4}' results/tuning/rerank_select.txt)"
WIN="$BASE+rerank@$DEPTH"
# Base strategies listed before the reranked one so LLM generations are cached first.
FINAL=$(echo "dense,bm25,hybrid,hyde,hyde_hybrid,fusion_hybrid,$BASE,$WIN" | tr ',' '\n' | awk '!seen[$0]++' | paste -sd,)
python scripts/run_retrieval_eval.py --split test --strategies "$FINAL" --reranker "$RERANKER" \
  --out results/retrieval_rerank_scifact_test.json
python scripts/run_retrieval_eval.py --dataset nfcorpus --split test --hyde-prompt question \
  --strategies "$FINAL" --reranker "$RERANKER" --out results/retrieval_rerank_nfcorpus_test.json
echo RERANK_DONE
