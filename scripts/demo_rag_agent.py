"""Demo: the rebuilt 2025 RAG agent (LangGraph + retriever tool) over SciFact.

Saves every message (questions, tool calls, retrieved passage ids, answers) to
results/rag_demo.json so an end-to-end trace exists as evidence.

Usage: python scripts/demo_rag_agent.py --model qwen2.5:7b --strategy hyde
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scalable_agents.data import download_beir, load_beir  # noqa: E402
from scalable_agents.retrieval import BM25Index, DenseIndex, Retriever  # noqa: E402

QUESTIONS = [
    "Does vitamin D supplementation reduce the risk of fractures?",
    "What is known about the role of microRNAs in cancer?",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--strategy", default="hyde", choices=Retriever.STRATEGIES)
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args()

    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    from scalable_agents.agent import build_rag_agent
    from scalable_agents.models import CachedGenerator, ollama_chat_model, sentence_transformer_embedder

    data = load_beir(download_beir("scifact", "data"))
    ids = list(data.corpus)
    dense = DenseIndex(sentence_transformer_embedder()).build(ids, [data.corpus[d] for d in ids])
    bm25 = BM25Index().build(ids, [data.corpus[d] for d in ids])
    gen = CachedGenerator(args.model, f"results/cache/{args.model.replace(':', '_')}.jsonl")
    retriever = Retriever(dense, bm25, gen)

    def search(query: str):
        return [(d, data.corpus[d][:700]) for d in retriever.search(query, args.k, args.strategy)]

    agent = build_rag_agent(ollama_chat_model(args.model), search)
    transcript = []
    for q in QUESTIONS:
        msgs = agent.invoke({"messages": [HumanMessage(content=q)]},
                            config={"recursion_limit": 11})["messages"]
        steps = []
        for m in msgs:
            if isinstance(m, AIMessage) and m.tool_calls:
                steps.append({"type": "tool_call", "calls": [c["args"] for c in m.tool_calls]})
            elif isinstance(m, ToolMessage):
                steps.append({"type": "observation",
                              "passage_ids": [line[1:line.index("]")] for line in m.content.split("\n---\n")
                                              if line.startswith("[")]})
            elif isinstance(m, AIMessage):
                steps.append({"type": "answer", "text": m.content})
        transcript.append({"question": q, "steps": steps})
        print(f"\nQ: {q}")
        for s in steps:
            print("  ", json.dumps(s)[:400])

    os.makedirs("results", exist_ok=True)
    with open("results/rag_demo.json", "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "strategy": args.strategy, "k": args.k,
                   "transcript": transcript}, f, indent=2)
    print("\nwrote results/rag_demo.json")


if __name__ == "__main__":
    main()
