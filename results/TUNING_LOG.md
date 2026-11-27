# Tuning log

Every change made after seeing a result, in order, with the evidence that motivated it.
Rule followed throughout: changes are judged on **dev data only** (agent: 12 dev tasks;
retrieval: SciFact *train* queries). The **test** sets (30 agent tasks, 300 SciFact test
queries) are run once with the final configuration and reported as-is.

## Agent (ReAct, LangGraph)

| # | Change | Data | Result | Kept? |
|---|---|---|---|---|
| 0 | First run, qwen2.5:7b, prompt said "digits only" | test (30) | 16/30 = 53.3% | - |
| 1 | **Bug fix**: "digits only" made the model drop decimal points (36.00 -> `3600`, 60.0688 -> `600688`). Prompt now asks for plain decimal form. | dev (12) | 7/12 = 58.3% | yes (bug fix) |
| 2 | Retry "nudge" (max 2) when the model stops without a FINAL ANSWER line (it sometimes returned an empty reply mid-task) | dev | 9/12 = 75.0% | yes |
| 3 | Explicit "write a plan first / never guess" prompt + more detailed tool docs | dev | 7/12 = 58.3% | **no, reverted** |
| 4 | qwen3:8b (thinking model) | dev | not finished: too slow on a shared T4 | dropped |
| 5 | qwen2.5:14b vs qwen2.5:7b, winner picked by dev score (`select_agent_model.py`) | dev | see results | rule fixed before running |

Run 0 is kept in `results/tuning/agent_qwen2.5_7b_test_v1_digits_prompt.json`. Note that it
was a test-set run; the bug was found by reading its failed replies, which is a harness
fix, not tuning toward the test answers.

## Retrieval (SciFact)

Pre-declared before any run: baseline = dense (all-MiniLM-L6-v2); primary metric = hit@5;
secondary = nDCG@10; LLM = qwen2.5:3b; HyDE: 1 passage, averaged with the query
embedding; RAG-Fusion: 3 rephrasings + original, RRF k=60, depth 100.

| # | Change | Data | Result | Kept? |
|---|---|---|---|---|
| 0 | First attempt failed: BEIR download server timed out | - | - | added Hugging Face mirror fallback |
| 1 | All 7 strategies with default settings | train (200 queries) | see `results/tuning/retrieval_train_default.json` | - |
| 2 | Final run, all strategies, default settings | test (300) | `results/retrieval.json` | reported |

## Infrastructure problems hit on Colab (not tuning, but part of the story)

- llama.cpp CUDA build takes 30+ min on Colab's 2 CPU cores; moved to a detached background job.
- `cmd &` inside a notebook cell still blocks the cell (the job keeps the cell's terminal); fixed with `scripts/bg.py` (new session, no terminal).
- Running agent evals while the retrieval job used a different model made Ollama swap models in and out of GPU memory; everything slowed down 5-10x. Fix: one GPU job at a time (`scripts/run_final_queue.sh`), `OLLAMA_NUM_PARALLEL=4`.
