#!/usr/bin/env bash
# Runs the remaining experiments one after another (one GPU job at a time).
# Launch detached:  python scripts/bg.py /content/final.log bash scripts/run_final_queue.sh
set -x
cd "$(dirname "$0")/.."

# 0. let a running train-split tuning job finish first
while pgrep -f '[r]un_retrieval_eval.py --split train' > /dev/null; do sleep 20; done

# keep the first (buggy 'digits only' prompt) test result as evidence before it is overwritten
mkdir -p results/tuning
[ -f results/agent_qwen2.5_7b.json ] && [ ! -f results/tuning/agent_qwen2.5_7b_test_v1_digits_prompt.json ] && \
  mv results/agent_qwen2.5_7b.json results/tuning/agent_qwen2.5_7b_test_v1_digits_prompt.json

# 1. retrieval: final TEST run with the pre-declared settings
python scripts/run_retrieval_eval.py --split test --workers 4 \
  --strategies dense,bm25,hybrid,hyde,rag_fusion,hyde_hybrid,fusion_hybrid

# 2. agent: 7b test (baseline config), 14b dev, then test the dev winner
python scripts/run_agent_eval.py --model qwen2.5:7b --split test
ollama pull qwen2.5:14b > /dev/null 2>&1
python scripts/run_agent_eval.py --model qwen2.5:14b --split dev
BEST=$(python scripts/select_agent_model.py qwen2.5:7b qwen2.5:14b)
echo "dev-selected agent model: $BEST"
if [ "$BEST" != "qwen2.5:7b" ]; then
  python scripts/run_agent_eval.py --model "$BEST" --split test
fi

# 3. RAG agent demo trace, 4. ONNX (CPU), 5. summary
python scripts/demo_rag_agent.py --model qwen2.5:7b --strategy hyde
ollama stop qwen2.5:14b; ollama stop qwen2.5:7b
python scripts/run_onnx_eval.py
python scripts/summarize.py
echo QUEUE_DONE
