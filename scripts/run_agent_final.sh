#!/usr/bin/env bash
# Final agent measurement on the 30 FRESH tasks (generated after all tuning).
# Decision rule, fixed before running: use the current prompt/tool docs if they score
# >= 9/12 on dev (the previous best); otherwise fall back to commit a71ea82's agent.py.
set -x
cd "$(dirname "$0")/.."
while pgrep -f '[r]un_final_queue.sh' > /dev/null; do sleep 20; done

python scripts/run_agent_eval.py --model qwen2.5:7b --split dev \
  --out results/tuning/agent_qwen2.5_7b_dev_v4_prompt_fix.json
DEV=$(python -c "import json; print(json.load(open('results/tuning/agent_qwen2.5_7b_dev_v4_prompt_fix.json'))['summary']['strict_success_count'])")
echo "dev score with prompt fix: $DEV/12"
if [ "$DEV" -ge 9 ]; then
  echo "using current agent.py"
else
  echo "falling back to agent.py from a71ea82"
  git show a71ea82:scalable_agents/agent.py > scalable_agents/agent.py
fi
python scripts/run_agent_eval.py --model qwen2.5:7b --split fresh
git checkout -- scalable_agents/agent.py
python scripts/summarize.py
echo AGENT_FINAL_DONE
