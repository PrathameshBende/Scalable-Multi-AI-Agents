#!/usr/bin/env bash
# Round 2 of agent improvements (protocol in results/TUNING_LOG.md, written before running).
# 1. dev_all (42 tasks), 5 votes, once per shipping tool -> six configs scored.
# 2. scripts/select_round2.py picks one config by the fixed rule.
# 3. That config runs ONCE on fresh2 (30 new tasks) -> the reported number.
# 4. Secondary, labelled as already-seen data: same config on the old test and fresh sets.
set -x
cd "$(dirname "$0")/.."
M=qwen2.5:7b
for TOOL in weight product; do
  python scripts/run_agent_eval.py --model $M --split dev_all --shipping $TOOL --votes 5 \
    --out results/tuning/agent_round2_dev_all_${TOOL}_v5.json
done
CHOICE=$(python scripts/select_round2.py | tee /dev/stderr | awk '/^CHOICE/ {print $2, $3}')
read TOOL VOTES <<< "$CHOICE"
python scripts/run_agent_eval.py --model $M --split fresh2 --shipping $TOOL --votes $VOTES \
  --out results/agent_round2_fresh2.json
python scripts/run_agent_eval.py --model $M --split fresh --shipping $TOOL --votes $VOTES \
  --out results/agent_round2_fresh_seen.json
python scripts/run_agent_eval.py --model $M --split test --shipping $TOOL --votes $VOTES \
  --out results/agent_round2_test_seen.json
python scripts/summarize.py
echo ROUND2_DONE
