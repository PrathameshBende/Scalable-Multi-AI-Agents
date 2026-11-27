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
| 5 | qwen2.5:14b vs qwen2.5:7b, winner picked by dev score (`select_agent_model.py`) | dev | 14b: 5/11 before it got stuck in a runaway generation on the last task; 7b kept | rule fixed before running |
| 6 | **Final test run**, qwen2.5:7b, config from #2 | test (30) | **19/30 = 63.3%** (CI 46-78%); shipping 0/6 | reported |
| 7 | Prompt shows a placeholder instead of the copyable "1234.56" (14b copied it); shipping tool doc says weight is the TOTAL weight | dev | 8/12 = 66.7%, below the pre-set bar of 9/12 | **no**, rule fell back to #2 |
| 8 | **Final run on 30 fresh template tasks** generated after all tuning, config from #2 | fresh (30) | **18/30 = 60.0%** (CI 42-75%) | reported |
| - | Reply length capped at 1024 tokens (after the 14b runaway) | - | no effect on normal answers | yes |

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
| 1 | All 7 strategies with default settings | train (200 queries) | dense hit@5 0.765; HyDE 0.800 (+3.5 pts, p=0.04); best nDCG@10: fusion_hybrid 0.712 vs 0.675 | - |
| 2 | Final run, all strategies, default settings (no further tuning was done) | test (300) | dense 0.753 / 0.645; hyde_hybrid **0.780 / 0.694** (hit@5 / nDCG@10): +2.7 pts (+3.5%) hit@5, p=0.24; **+7.6% nDCG@10** | reported |

## Compression

| # | Run | Result |
|---|---|---|
| 1 | ONNX export of all-MiniLM-L6-v2 (fp32) | identical retrieval to PyTorch (cosine 1.0000) |
| 2 | ONNX dynamic int8 | 90.4 -> 22.9 MB (-74.9%); nDCG@10 -0.97% relative; **hit@5 -3.0 pts** |
| 3 | GGUF (llama.cpp) | not finished: CUDA build too slow on Colab's 2 CPU cores |

## Infrastructure problems hit on Colab (not tuning, but part of the story)

- llama.cpp CUDA build takes 30+ min on Colab's 2 CPU cores; moved to a detached background job.
- `cmd &` inside a notebook cell still blocks the cell (the job keeps the cell's terminal); fixed with `scripts/bg.py` (new session, no terminal).
- Running agent evals while the retrieval job used a different model made Ollama swap models in and out of GPU memory; everything slowed down 5-10x. Fix: one GPU job at a time (`scripts/run_final_queue.sh`), `OLLAMA_NUM_PARALLEL=4`.
