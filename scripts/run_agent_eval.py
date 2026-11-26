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
from scalable_agents.tasks import TASKS, grade  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--max-steps", type=int, default=10, help="max agent->tools loops")
    ap.add_argument("--repeats", type=int, default=1, help="run every task this many times")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from langgraph.errors import GraphRecursionError

    from scalable_agents.agent import build_react_agent, count_tool_calls, store_tools
    from scalable_agents.models import environment_info, ollama_chat_model

    agent = build_react_agent(ollama_chat_model(args.model), store_tools())
    records = []
    for rep in range(args.repeats):
        for task in TASKS:
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
            g = grade(task, reply)
            tools_used = [c["name"] for m in msgs if isinstance(m, AIMessage) for c in m.tool_calls]
            tool_errors = sum(1 for m in msgs if isinstance(m, ToolMessage) and '"error"' in str(m.content))
            rec = {
                "repeat": rep, "id": task.id, "category": task.category, "question": task.question,
                "min_tools": task.min_tools, "tool_calls": count_tool_calls(msgs),
                "tools_used": tools_used, "tool_errors": tool_errors,
                "seconds": round(time.time() - t0, 2), "error": error,
                "reply": reply[-600:], **g,
            }
            records.append(rec)
            mark = "PASS" if g["strict"] else "fail"
            print(f"[{mark}] {task.id} expected={g['expected']:.4g} got={g['parsed']} "
                  f"tools={rec['tool_calls']} {rec['seconds']}s {error or ''}")

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
        "n_runs": n, "n_tasks": len(TASKS), "repeats": args.repeats,
        "strict_success": strict / n, "strict_success_count": strict,
        "strict_95ci": [lo, hi], "lenient_success": lenient / n,
        "by_category": {c: {"pass": p, "n": t, "rate": p / t} for c, (p, t) in by_cat.items()},
        "mean_tool_calls": sum(r["tool_calls"] for r in records) / n,
        "runs_without_any_tool_call": sum(1 for r in records if r["tool_calls"] == 0),
        "runs_with_tool_errors": sum(1 for r in records if r["tool_errors"]),
        "runs_crashed_or_step_limit": sum(1 for r in records if r["error"]),
        "mean_seconds": sum(r["seconds"] for r in records) / n,
    }
    out_path = args.out or f"results/agent_{args.model.replace(':', '_').replace('/', '_')}.json"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"experiment": "react_agent", "model": args.model, "max_steps": args.max_steps,
                   "summary": summary, "records": records, "env": environment_info()}, f, indent=2)
    print(f"\nstrict success {strict}/{n} = {100 * strict / n:.1f}% "
          f"(95% CI {100 * lo:.0f}-{100 * hi:.0f}%), lenient {100 * lenient / n:.1f}%")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
