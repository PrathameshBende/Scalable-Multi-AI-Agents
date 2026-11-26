"""Model compression: ONNX int8 for the embedding model, GGUF for the LLM.

ONNX path (embedder, all-MiniLM-L6-v2):
  PyTorch weights -> torch.onnx.export (fp32 graph) -> onnxruntime dynamic
  quantization (weights stored as int8, activations quantized on the fly).
  Evaluated by re-running the SciFact retrieval benchmark with each version.

GGUF path (LLM, Qwen2.5-0.5B-Instruct), driven by scripts/run_gguf_eval.py:
  HF safetensors -> llama.cpp convert_hf_to_gguf.py (f16 GGUF)
  -> llama-quantize (Q8_0, Q4_K_M) -> llama-perplexity on WikiText-2
  and HellaSwag accuracy. The parsers for llama.cpp's output live here so they
  can be unit-tested against sample output.
"""

from __future__ import annotations

import os
import re
from typing import Sequence

import numpy as np


def file_size_mb(path: str) -> float:
    return os.path.getsize(path) / 1e6


# --------------------------------------------------------------------------- ONNX
def export_embedder_onnx(model_name: str, out_path: str, opset: int = 17) -> str:
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval()
    sample = tok(["a sample sentence for tracing"], return_tensors="pt")
    names = [n for n in ("input_ids", "attention_mask", "token_type_ids") if n in sample]
    dyn = {n: {0: "batch", 1: "seq"} for n in names}
    dyn["last_hidden_state"] = {0: "batch", 1: "seq"}
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    kwargs = dict(
        input_names=names, output_names=["last_hidden_state"], dynamic_axes=dyn,
        opset_version=opset, do_constant_folding=True,
    )

    class Wrapper(torch.nn.Module):  # return only last_hidden_state
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, *args):
            return self.m(**dict(zip(names, args))).last_hidden_state

    with torch.no_grad():
        try:  # TorchScript exporter: stable for BERT-style models
            torch.onnx.export(Wrapper(model), tuple(sample[n] for n in names), out_path,
                              dynamo=False, **kwargs)
        except TypeError:  # older torch without the dynamo argument
            torch.onnx.export(Wrapper(model), tuple(sample[n] for n in names), out_path, **kwargs)
    return out_path


def quantize_onnx_int8(fp32_path: str, int8_path: str) -> str:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(fp32_path, int8_path, weight_type=QuantType.QInt8)
    return int8_path


def mean_pool(last_hidden: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mean over real tokens only (what all-MiniLM-L6-v2 uses), then L2-normalise."""
    m = mask[..., None].astype(np.float32)
    summed = (last_hidden * m).sum(axis=1)
    counts = np.clip(m.sum(axis=1), 1e-9, None)
    emb = summed / counts
    return emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12, None)


class OnnxEmbedder:
    """list[str] -> normalised embeddings, using onnxruntime on CPU."""

    def __init__(self, onnx_path: str, tokenizer_name: str, max_length: int = 256,
                 batch_size: int = 64, threads: int | None = None):
        import onnxruntime as ort
        from transformers import AutoTokenizer

        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(onnx_path, opts, providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.session.get_inputs()}
        self.tok = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.batch_size = batch_size

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), self.batch_size):
            enc = self.tok(list(texts[i:i + self.batch_size]), padding=True, truncation=True,
                           max_length=self.max_length, return_tensors="np")
            feed = {k: v.astype(np.int64) for k, v in enc.items() if k in self.inputs}
            hidden = self.session.run(["last_hidden_state"], feed)[0]
            out.append(mean_pool(hidden, enc["attention_mask"]))
        return np.concatenate(out, axis=0)


# --------------------------------------------------------------------------- GGUF
_PPL = re.compile(r"Final estimate:\s*PPL\s*=\s*([\d.]+)\s*\+/-\s*([\d.]+)")
_HS_LINE = re.compile(r"^\s*(\d+)\s+([\d.]+)%?(?:\s+\[([\d.]+)%?,\s*([\d.]+)%?\])?\s*$")


def parse_perplexity(output: str) -> dict:
    m = _PPL.search(output)
    if not m:
        raise ValueError("no 'Final estimate: PPL = ...' line in llama-perplexity output")
    return {"ppl": float(m.group(1)), "ppl_stderr": float(m.group(2))}


def parse_hellaswag(output: str) -> dict:
    """llama-perplexity --hellaswag prints a running accuracy per task; take the last line."""
    last = None
    for line in output.splitlines():
        m = _HS_LINE.match(line)
        if m:
            last = m
    if last is None:
        raise ValueError("no HellaSwag score lines found")
    res = {"tasks": int(last.group(1)), "acc": float(last.group(2))}
    if last.group(3):
        res["ci_low"], res["ci_high"] = float(last.group(3)), float(last.group(4))
    return res
