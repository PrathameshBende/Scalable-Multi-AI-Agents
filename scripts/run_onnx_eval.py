"""Export the embedder to ONNX, quantize to int8, and measure size, speed and retrieval quality.

Compared on the SciFact benchmark (same 300 queries as run_retrieval_eval.py):
  pytorch_fp32  : sentence-transformers on CPU (reference quality and speed)
  onnx_fp32     : exported graph, onnxruntime CPU
  onnx_int8     : dynamic int8 quantization, onnxruntime CPU
All three run on CPU so the speed numbers are comparable.

Usage: python scripts/run_onnx_eval.py
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from scalable_agents.data import download_beir, load_beir  # noqa: E402
from scalable_agents.metrics import evaluate_run  # noqa: E402
from scalable_agents.retrieval import DenseIndex  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--workdir", default="work/onnx")
    ap.add_argument("--speed-docs", type=int, default=1000, help="docs used for the timing test")
    ap.add_argument("--out", default="results/onnx.json")
    args = ap.parse_args()

    from huggingface_hub import snapshot_download
    from sentence_transformers import SentenceTransformer

    from scalable_agents.compress import (
        OnnxEmbedder, export_embedder_onnx, file_size_mb, quantize_onnx_int8,
    )
    from scalable_agents.models import environment_info

    data = load_beir(download_beir("scifact", "data"))
    doc_ids = list(data.corpus)
    docs = [data.corpus[d] for d in doc_ids]
    qids = list(data.queries)

    fp32 = export_embedder_onnx(args.model, os.path.join(args.workdir, "model_fp32.onnx"))
    int8 = quantize_onnx_int8(fp32, os.path.join(args.workdir, "model_int8.onnx"))
    # Size of the original PyTorch weights file, for reference.
    snap = snapshot_download(args.model, allow_patterns=["*.safetensors", "*.bin", "*.json"])
    weights = (glob.glob(os.path.join(snap, "*.safetensors")) or glob.glob(os.path.join(snap, "*.bin")))[0]

    st = SentenceTransformer(args.model, device="cpu")
    variants = {
        "pytorch_fp32": lambda t: st.encode(t, batch_size=64, normalize_embeddings=True,
                                            convert_to_numpy=True),
        "onnx_fp32": OnnxEmbedder(fp32, args.model, max_length=st.max_seq_length),
        "onnx_int8": OnnxEmbedder(int8, args.model, max_length=st.max_seq_length),
    }
    sizes = {"pytorch_fp32": file_size_mb(weights), "onnx_fp32": file_size_mb(fp32),
             "onnx_int8": file_size_mb(int8)}

    results, doc_vecs = {}, {}
    speed_sample = docs[: args.speed_docs]
    for name, embed in variants.items():
        embed(speed_sample[:64])  # warm-up
        t0 = time.time()
        embed(speed_sample)
        docs_per_sec = len(speed_sample) / (time.time() - t0)
        index = DenseIndex(embed).build(doc_ids, docs)
        doc_vecs[name] = index.matrix
        run = {q: index.search(data.queries[q], 10) for q in qids}
        results[name] = {"size_mb": sizes[name], "cpu_docs_per_sec": docs_per_sec,
                         "metrics": evaluate_run(run, data.qrels)}
        m = results[name]["metrics"]
        print(f"{name:>13}: {sizes[name]:6.1f} MB  {docs_per_sec:7.1f} docs/s  "
              f"hit@5={m['hit@5']:.4f} ndcg@10={m['ndcg@10']:.4f}")

    ref = results["pytorch_fp32"]
    for name in ("onnx_fp32", "onnx_int8"):
        r = results[name]
        cos = np.sum(doc_vecs[name] * doc_vecs["pytorch_fp32"], axis=1)
        r["vs_pytorch"] = {
            "size_reduction_pct": 100 * (1 - r["size_mb"] / ref["size_mb"]),
            "speedup_x": r["cpu_docs_per_sec"] / ref["cpu_docs_per_sec"],
            "ndcg@10_abs_points": 100 * (r["metrics"]["ndcg@10"] - ref["metrics"]["ndcg@10"]),
            "ndcg@10_rel_pct": 100 * (r["metrics"]["ndcg@10"] / ref["metrics"]["ndcg@10"] - 1),
            "hit@5_abs_points": 100 * (r["metrics"]["hit@5"] - ref["metrics"]["hit@5"]),
            "hit@5_rel_pct": 100 * (r["metrics"]["hit@5"] / ref["metrics"]["hit@5"] - 1),
            "mean_cosine_to_pytorch_embeddings": float(cos.mean()),
            "min_cosine_to_pytorch_embeddings": float(cos.min()),
        }
        v = r["vs_pytorch"]
        print(f"{name} vs pytorch: size {-v['size_reduction_pct']:+.1f}%, speed x{v['speedup_x']:.2f}, "
              f"ndcg@10 {v['ndcg@10_rel_pct']:+.2f}% rel, hit@5 {v['hit@5_abs_points']:+.2f} pts, "
              f"mean cos {v['mean_cosine_to_pytorch_embeddings']:.4f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"experiment": "onnx_embedder", "model": args.model, "n_queries": len(qids),
                   "speed_docs": len(speed_sample), "cpu_threads": os.cpu_count(),
                   "variants": results, "env": environment_info()}, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
