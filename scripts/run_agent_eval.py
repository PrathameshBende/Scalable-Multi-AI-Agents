"""Run the ReAct agent on the 30 tasks and write results/agent_<model>.json.

Headline metric: strict task success = the 'FINAL ANSWER:' number is correct
(within 1%). Reported with a 95% Wilson interval, because n = 30 is small.

Usage (Colab, after `ollama pull <model>`):
  python scripts/run_agent_eval.py --model qwen2.5:7b
  python scripts/run_agent_eval.py --model qwen2.5:3b
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scalable_agents.metrics import wilson_interval  # noqa: E402
from scalable_agents.tasks import (  # noqa: E402
    DEV2_TASKS, DEV_TASKS, FRESH2_TASKS, FRESH_TASKS, TASKS, grade, majority_vote, parse_final_answer,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--max-steps", type=int, default=12, help="max agent->tools loops")
    ap.add_argument("--max-nudges", type=int, default=2, help="0 disables the retry nudge")
    ap.add_argument("--repeats", type=int, default=1, help="run every task this many times")
    ap.add_argument("--shipping", default="weight", choices=["weight", "product"],
                    help="shipping tool interface: 'weight' (v1) or 'product' (v2, tool multiplies)")
    ap.add_argument("--votes", type=int, default=1, help="self-consistency: runs per task, majority answer wins")
    ap.add_argument("--vote-temperature", type=float, default=0.7)
    ap.add_argument("--split", default="test", choices=["dev", "dev2", "dev_all", "test", "fresh", "fresh2"],
                    help="'dev' (12) for trying changes; 'test' (30, seen while debugging the harness); "
                         "'fresh' (30 template tasks generated after tuning) for the final number; "
                         "round 2: 'dev2'/'dev_all' for tuning, 'fresh2' for the final number")
    ap.add_argument("--show-failures", action="store_true", help="print the full reply of failed tasks")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from langgraph.errors import GraphRecursionError

    from scalable_agents.agent import build_react_agent, count_tool_calls, store_tools
    from scalable_agents.models import environment_info, ollama_chat_model

    tasks = {"test": TASKS, "dev": DEV_TASKS, "dev2": DEV2_TASKS, "dev_all": DEV_TASKS + DEV2_TASKS,
             "fresh": FRESH_TASKS, "fresh2": FRESH2_TASKS}[args.split]
    tools = store_tools(shipping=args.shipping)
    # Vote 0 is the greedy (temperature 0) run, exactly the single-run agent. Votes 1.. are
    # seeded samples at temperature 0.7 (self-consistency, Wang et al. 2022).
    agents = [build_react_agent(ollama_chat_model(args.model), tools, max_nudges=args.max_nudges)]
    for v in range(1, args.votes):
        llm = ollama_chat_model(args.model, temperature=args.vote_temperature, seed=v)
        agents.append(build_react_agent(llm, tools, max_nudges=args.max_nudges))

    def run_once(agent, task) -> dict:
        t0 = time.time()
        error = None
        try:
            state = agent.invoke(
                {"messages": [HumanMessage(content=task.question)]},
                config={"recursion_limit": 2 * args.max_steps + 1},
            )
            msgs = state["messages"]
        except GraphRecursionError:
            msgs, error = [], f"exceeded {args.max_steps} steps"
        except Exception as e:  # keep going; a crash is a failed task
            msgs, error = [], f"{type(e).__name__}: {e}"
        reply = msgs[-1].content if msgs and isinstance(msgs[-1], AIMessage) else ""
        return {
            "reply": reply, "parsed": parse_final_answer(reply), "error": error,
            "tool_calls": count_tool_calls(msgs),
            "tools_used": [c["name"] for m in msgs if isinstance(m, AIMessage) for c in m.tool_calls],
            "tool_errors": sum(1 for m in msgs if isinstance(m, ToolMessage) and '"error"' in str(m.content)),
            "seconds": time.time() - t0,
        }

    records = []
    for rep in range(args.repeats):
        for task in tasks:
            runs = [run_once(a, task) for a in agents]
            chosen = runs[majority_vote([r["parsed"] for r in runs])]
            g = grade(task, chosen["reply"])
            rec = {
                "repeat": rep, "id": task.id, "category": task.category, "question": task.question,
                "min_tools": task.min_tools, "tool_calls": sum(r["tool_calls"] for r in runs),
                "tools_used": chosen["tools_used"], "tool_errors": sum(r["tool_errors"] for r in runs),
                "seconds": round(sum(r["seconds"] for r in runs), 2), "error": chosen["error"],
                "reply": chosen["reply"][-600:], **g,
            }
            if args.votes > 1:
                rec["votes"] = [r["parsed"] for r in runs]
                rec["greedy_strict"] = grade(task, runs[0]["reply"])["strict"]
            records.append(rec)
            mark = "PASS" if g["strict"] else "fail"
            votes = f" votes={rec['votes']}" if args.votes > 1 else ""
            print(f"[{mark}] {task.id} expected={g['expected']:.4g} got={g['parsed']}{votes} "
                  f"tools={rec['tool_calls']} {rec['seconds']}s {rec['error'] or ''}", flush=True)
            if args.show_failures and not g["strict"]:
                print(f"    tools used: {rec['tools_used']}\n    reply: {chosen['reply'][-400:]!r}", flush=True)

    n = len(records)
    strict = sum(r["strict"] for r in records)
    lenient = sum(r["lenient"] for r in records)
    by_cat: dict[str, list[int]] = {}
    for r in records:
        by_cat.setdefault(r["category"], [0, 0])
        by_cat[r["category"]][0] += int(r["strict"])
        by_cat[r["category"]][1] += 1
    lo, hi = wilson_interval(strict, n)
    summary = {
        "n_runs": n, "n_tasks": len(tasks), "split": args.split, "repeats": args.repeats,
        "shipping_tool": args.shipping, "votes": args.votes,
        "strict_success": strict / n, "strict_success_count": strict,
        "strict_95ci": [lo, hi], "lenient_success": lenient / n,
        "by_category": {c: {"pass": p, "n": t, "rate": p / t} for c, (p, t) in by_cat.items()},
        "mean_tool_calls": sum(r["tool_calls"] for r in records) / n,
        "runs_without_any_tool_call": sum(1 for r in records if r["tool_calls"] == 0),
        "runs_with_tool_errors": sum(1 for r in records if r["tool_errors"]),
        "runs_crashed_or_step_limit": sum(1 for r in records if r["error"]),
        "mean_seconds": sum(r["seconds"] for r in records) / n,
    }
    if args.votes > 1:
        summary["greedy_only_strict_count"] = sum(r["greedy_strict"] for r in records)
    suffix = "" if args.split == "test" else f"_{args.split}"
    out_path = args.out or f"results/agent_{args.model.replace(':', '_').replace('/', '_')}{suffix}.json"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"experiment": "react_agent", "model": args.model, "shipping_tool": args.shipping, "votes": args.votes, "max_steps": args.max_steps, "max_nudges": args.max_nudges,
                   "summary": summary, "records": records, "env": environment_info()}, f, indent=2)
    print(f"\nstrict success {strict}/{n} = {100 * strict / n:.1f}% "
          f"(95% CI {100 * lo:.0f}-{100 * hi:.0f}%), lenient {100 * lenient / n:.1f}%")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
