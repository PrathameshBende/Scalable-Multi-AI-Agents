# Scalable-Multi-AI-Agents

RAG and ReAct agents built with LangGraph, advanced retrieval (HyDE, RAG-Fusion,
hybrid search), and model compression (ONNX int8, GGUF), each with a
reproducible evaluation.

## History (be precise about this)

| Version | When | What |
|---|---|---|
| v1 | May-Jul 2025, WnCC Summer of Code | `legacy/SOC_Final_Project.ipynb`: one LangGraph RAG agent (Mistral via Ollama, Chroma, PDF Q&A). No evaluation. |
| v2 | Nov 2026 rebuild | This package: v1's agent rebuilt, plus HyDE/RAG-Fusion, a tool-using ReAct agent, ONNX/GGUF compression, and benchmarks for all of them. |

## What is here

```
scalable_agents/
  retrieval.py   dense + BM25 indexes; strategies: dense, bm25, hyde, rag_fusion, hybrid
  fusion.py      Reciprocal Rank Fusion, multi-query parsing
  metrics.py     hit@k, recall@k, MRR, nDCG, Wilson CI, McNemar test
  agent.py       LangGraph ReAct loop (agent <-> tools), RAG agent, store tools
  tools.py       7 pure-Python tools over fictional store data
  tasks.py       30 multi-step tasks, answers computed from the tools, grader
  compress.py    ONNX export / int8 quantization / ONNX embedder; llama.cpp output parsers
  data.py        BEIR SciFact loader
scripts/         one script per experiment; each writes results/<name>.json
tests/           48 unit tests, no models needed (fake LLM, fake embedder)
notebooks/colab_run_all.ipynb   runs everything on a free Colab T4
legacy/          the original 2025 notebook (outputs removed)
```

## Experiments and what they measure

| Script | Question | Data | Metric (pre-declared) |
|---|---|---|---|
| `run_retrieval_eval.py` | Do HyDE / RAG-Fusion / hybrid beat plain dense retrieval? | BEIR SciFact, 5,183 abstracts, 300 test claims | hit@5 vs dense baseline (+ nDCG@10), McNemar p-value |
| `run_agent_eval.py` | How often does the ReAct agent solve multi-step tool tasks? | 30 tasks in `tasks.py`, 2-5 tool calls each | strict success (FINAL ANSWER within 1%), 95% Wilson CI |
| `run_onnx_eval.py` | What does ONNX int8 cost in quality, and what does it save? | SciFact retrieval with each embedder version | size MB, CPU docs/s, nDCG@10 and hit@5 change |
| `run_gguf_eval.py` | Same question for the LLM | Qwen2.5-0.5B-Instruct, WikiText-2, HellaSwag | size MB, perplexity change %, HellaSwag accuracy change |

## Run it

Unit tests (any machine, CPU):

```
pip install numpy pytest rank-bm25 langgraph langchain-core
python -m pytest -q
```

Experiments (Colab T4): `python scripts/make_zip.py`, open `notebooks/colab_run_all.ipynb`
in Colab, Run all, upload `dist/project.zip` when asked, then unzip the downloaded
`results.zip` into `results/`. `python scripts/summarize.py` rebuilds `results/SUMMARY.md`.

## Results (Colab T4, 2026-11-27)

Full tables: `results/SUMMARY.md`. Every change made after seeing a number: `results/TUNING_LOG.md`.

| Experiment | Headline (honest) |
|---|---|
| Retrieval, SciFact test (300 q) | HyDE + BM25 fused with RRF: nDCG@10 0.645 -> 0.694 (**+7.6%**), hit@5 0.753 -> 0.780 (+2.7 pts, p = 0.24, not significant) |
| ReAct agent, qwen2.5:7b | **63.3%** (19/30) on the test tasks; **60.0%** (18/30) on 30 fresh tasks generated after tuning |
| ONNX int8 embedder | 90.4 -> 22.9 MB (**-74.9%**); nDCG@10 -0.97% relative, hit@5 -3.0 pts |
| GGUF | not finished (llama.cpp CUDA build too slow on Colab's 2 CPU cores) |

## Known limitations

- 30 agent tasks is a small set: the 95% interval is roughly +-10 points.
- Agent tasks use fictional data written for this repo; they test tool use and
  arithmetic chaining, not open-ended reasoning.
- SciFact is one domain (biomedical claims); HyDE's benefit varies by domain.
- Dense index is brute-force numpy (exact, fine for ~10^5 docs); larger corpora need ANN (FAISS/HNSW).
- LLM generations use temperature 0 and one sample; HyDE's paper averages several samples.
