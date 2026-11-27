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

### Round 2 (agent improvements), protocol written before any round-2 run

Motivation: shipping was the weakest category (0/6 test, 3/8 fresh). The model has to
multiply the unit weight by the quantity before calling `shipping_cost`, and sometimes
passed the unit weight. A longer chain also multiplies the chance of one wrong step.

Changes under test (code: `store_tools(shipping=...)`, `run_agent_eval.py --votes`):
- **Shipping tool v2**: `shipping_cost(region, product, quantity)`. The tool looks up the
  weight and multiplies. The questions and expected answers are unchanged.
- **Self-consistency**: run each task N times (vote 0 greedy, votes 1.. at temperature 0.7
  with seeds 1..), take the FINAL ANSWER most runs agree on (ties -> greedy run).

Data: all earlier sets have now been looked at, so round 2 uses new ones (`tasks.py`):
- `DEV2_TASKS` (30, seed 7) + the old 12 dev tasks = **dev_all, 42 tasks**, for choosing.
- `FRESH2_TASKS` (30, seed 2027), generated before any round-2 run, **used once**.

Rule (`scripts/select_round2.py`): score six configs {weight, product} x {1, 3, 5 votes}
on dev_all; highest strict count wins; ties -> fewer votes, then the old weight tool.
The winner runs once on fresh2 (reported). It also runs on the old test and fresh sets,
reported as secondary and labelled "already seen". Model stays qwen2.5:7b. The prompt
stays at the round-1 config (commit a71ea82). Script: `scripts/run_agent_round2.sh`.

| # | Run | Data | Result | Kept? |
|---|---|---|---|---|
| R2-1 | six configs (Kaggle T4; Colab's free GPU quota had run out) | dev_all (42) | weight tool: 27 / 28 / 28 with 1 / 3 / 5 votes; **product tool: 35 / 35 / 35** | product tool, 1 vote (rule) |
| R2-2 | chosen config, final | fresh2 (30) | **26/30 = 86.7%** (CI 70-95%); shipping 8/8 | reported |
| R2-3 | chosen config, secondary | fresh (seen) / test (seen) | 25/30 = 83.3% (was 60.0%) / 21/30 = 70.0% (was 63.3%) | reported, labelled seen |

What round 2 showed:
- The tool redesign did the work (+8 of 42 on dev); voting added at most 1. The weight-tool
  errors were systematic (e.g. dev03: all 5 votes passed the quantity 3 as the weight), and
  majority voting cannot fix an error every sample makes.
- New failure mode from the easier tool: on plain "total price" questions the model sometimes
  calls `shipping_cost` and adds shipping nobody asked for (G07, A01, A04, F03, F08). This is
  why lookup on the test set fell from 10/10 to 8/10. Not fixed: fresh2 is now used, so a fix
  would need a round 3 with new data.
- Round 2 ran on Kaggle, round 1 on Colab (both T4, same model and seed); GPU generation is not
  bit-exact across machines, so small differences on the seen sets are partly noise.
- Harness bug, no effect on results: `run_agent_round2.sh` piped through `tee /dev/stderr`,
  which truncated the console log; the result JSONs were unaffected. Fixed afterwards.

## Retrieval (SciFact)

Pre-declared before any run: baseline = dense (all-MiniLM-L6-v2); primary metric = hit@5;
secondary = nDCG@10; LLM = qwen2.5:3b; HyDE: 1 passage, averaged with the query
embedding; RAG-Fusion: 3 rephrasings + original, RRF k=60, depth 100.

| # | Change | Data | Result | Kept? |
|---|---|---|---|---|
| 0 | First attempt failed: BEIR download server timed out | - | - | added Hugging Face mirror fallback |
| 1 | All 7 strategies with default settings | train (200 queries) | dense hit@5 0.765; HyDE 0.800 (+3.5 pts, p=0.04); best nDCG@10: fusion_hybrid 0.712 vs 0.675 | - |
| 2 | Final run, all strategies, default settings (no further tuning was done) | test (300) | dense 0.753 / 0.645; hyde_hybrid **0.780 / 0.694** (hit@5 / nDCG@10): +2.7 pts (+3.5%) hit@5, p=0.24; **+7.6% nDCG@10** | reported |

### Round 2 (cross-encoder reranking), protocol written before any reranking run

Motivation: round 1 reached +2.7 pts hit@5 / +7.6% nDCG@10 over dense. HyDE and fusion
mostly reorder documents near the top; a cross-encoder reads query and document together
and can also pull a relevant document up from rank 20-100 into the top 5.

Change under test: `<base>+rerank@N` re-sorts the base strategy's top N by a CrossEncoder
score for (original query, document). Code: `Retriever.reranked`, `CrossEncoderReranker`.

Selection on SciFact **train** (first 200 queries, as in round 1), `scripts/select_rerank.py`:
- grid: base {dense, hybrid, hyde_hybrid, fusion_hybrid} x depth {20, 50, 100} x reranker
  {cross-encoder/ms-marco-MiniLM-L-6-v2, BAAI/bge-reranker-base}: 24 configs
- rule: highest hit@5 (the pre-declared primary metric); ties -> nDCG@10, smaller depth,
  smaller reranker, cheaper base
- baseline unchanged: dense all-MiniLM-L6-v2. HyDE / fusion settings unchanged from round 1.

Final, each run once (`scripts/run_rerank.sh`):
- SciFact test (300 q). This is the **second** look at this test set (round 1 was the
  first); reported as such.
- NFCorpus test (323 q, nutrition/medical questions), never used before: a clean held-out
  check. HyDE prompt for questions ("scientific paper passage to answer the question",
  following the HyDE paper's prompts for question datasets), fixed before running. The
  RAG-Fusion prompt is reused unchanged (it says "claim").

| # | Run | Data | Result | Kept? |
|---|---|---|---|---|
| RR-1 | 24 reranking configs | SciFact train (200) | (pending) | |
| RR-2 | chosen config vs dense | SciFact test (300), 2nd look | (pending) | |
| RR-3 | chosen config vs dense | NFCorpus test (323), held out | (pending) | |

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
