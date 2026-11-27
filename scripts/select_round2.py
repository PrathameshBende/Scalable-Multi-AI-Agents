"""Round-2 agent config choice, by a rule fixed before any round-2 run.

Inputs: two dev_all runs (42 tasks) with 5 votes each, one per shipping tool. Each run
also contains the 1-vote (greedy, vote 0) and 3-vote (votes 0-2) results, so the two
runs cover all six configs {weight, product} x {1, 3, 5 votes}.

Rule: highest strict count on dev_all wins; ties go to fewer votes (cheaper), then to
the 'weight' tool (unchanged interface). Writes results/tuning/round2_choice.json.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scalable_agents.tasks import close, majority_vote  # noqa: E402


def score(records: list[dict], k: int) -> int:
    hits = 0
    for r in records:
        votes = r["votes"][:k]
        hits += close(votes[majority_vote(votes)], r["expected"])
    return hits


def main() -> None:
    d = "results/tuning"
    rows = []
    for tool in ("weight", "product"):
        path = os.path.join(d, f"agent_round2_dev_all_{tool}_v5.json")
        records = json.load(open(path, encoding="utf-8"))["records"]
        for k in (1, 3, 5):
            rows.append({"shipping": tool, "votes": k, "strict": score(records, k), "n": len(records)})
    order = {"weight": 0, "product": 1}
    best = max(rows, key=lambda r: (r["strict"], -r["votes"], -order[r["shipping"]]))
    for r in rows:
        mark = "  <- chosen" if r is best else ""
        print(f"shipping={r['shipping']:<8} votes={r['votes']}  {r['strict']}/{r['n']}{mark}")
    with open(os.path.join(d, "round2_choice.json"), "w", encoding="utf-8") as f:
        json.dump({"rule": __doc__.strip().splitlines()[-2:], "table": rows, "chosen": best}, f, indent=2)
    print(f"CHOICE {best['shipping']} {best['votes']}")


if __name__ == "__main__":
    main()
