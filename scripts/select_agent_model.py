"""Pick the agent model by DEV-set score only (never the test set).

Usage: python scripts/select_agent_model.py qwen2.5:7b qwen2.5:14b
Prints the model with the highest strict dev success; ties go to the model listed first
(the smaller/cheaper one). Models without a dev result are skipped.
"""

import json
import os
import sys

best, best_score = None, -1.0
for model in sys.argv[1:]:
    path = f"results/agent_{model.replace(':', '_').replace('/', '_')}_dev.json"
    if not os.path.exists(path):
        print(f"no dev result for {model}", file=sys.stderr)
        continue
    with open(path, encoding="utf-8") as f:
        score = json.load(f)["summary"]["strict_success"]
    print(f"{model}: dev strict success {score:.3f}", file=sys.stderr)
    if score > best_score:
        best, best_score = model, score
if best is None:
    sys.exit("no dev results found")
print(best)
