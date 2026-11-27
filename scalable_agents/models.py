"""Thin wrappers that create real models. Imported only by scripts, never by tests."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from typing import Callable

import numpy as np


def sentence_transformer_embedder(
    name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str | None = None,
    batch_size: int = 64,
) -> Callable[[list[str]], np.ndarray]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(name, device=device)

    def embed(texts: list[str]) -> np.ndarray:
        return model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=len(texts) > 1000,
        )

    return embed


def ollama_chat_model(model: str, temperature: float = 0.0, num_ctx: int = 4096, seed: int = 0,
                      num_predict: int = 1024):
    from langchain_ollama import ChatOllama

    # num_predict caps one reply; without it a model can loop for many minutes (qwen2.5:14b did).
    return ChatOllama(model=model, temperature=temperature, num_ctx=num_ctx, seed=seed,
                      num_predict=num_predict)


class CachedGenerator:
    """str -> str LLM call with an on-disk JSONL cache.

    The cache makes reruns cheap and lets a Colab run resume after a disconnect.
    Keyed on (model, prompt) so a model change never reuses old generations.
    """

    def __init__(self, model: str, cache_path: str, temperature: float = 0.0,
                 sample_temperature: float = 0.7):
        self.model = model
        self.cache_path = cache_path
        self._llm = ollama_chat_model(model, temperature=temperature)
        self._sample_temperature = sample_temperature
        self._lock = threading.Lock()
        self._cache: dict[str, str] = {}
        self.calls = 0
        if os.path.exists(cache_path):
            with open(cache_path, encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    self._cache[row["key"]] = row["output"]

    def _key(self, prompt: str, sample: int = 0) -> str:
        # sample 0 keeps the original key, so earlier cached generations stay valid.
        tag = "" if sample == 0 else f"\n#sample={sample}@T{self._sample_temperature}"
        return hashlib.sha256(f"{self.model}\n{prompt}{tag}".encode()).hexdigest()

    def __call__(self, prompt: str, sample: int = 0) -> str:
        """sample=0 is the greedy (temperature 0) answer; sample>=1 are seeded random samples."""
        key = self._key(prompt, sample)
        if key in self._cache:
            return self._cache[key]
        if sample == 0:
            out = self._llm.invoke(prompt).content.strip()
        else:
            llm = ollama_chat_model(self.model, temperature=self._sample_temperature, seed=sample)
            out = llm.invoke(prompt).content.strip()
        with self._lock:
            self.calls += 1
            self._cache[key] = out
            os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
            with open(self.cache_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "prompt": prompt, "output": out}) + "\n")
        return out


def environment_info() -> dict:
    """Library versions and hardware, saved with every result for reproducibility."""
    import platform
    from importlib import metadata

    info = {"python": platform.python_version(), "platform": platform.platform()}
    for pkg in (
        "numpy", "torch", "transformers", "sentence-transformers", "onnx", "onnxruntime",
        "langgraph", "langchain-core", "langchain-ollama", "rank-bm25",
    ):
        try:
            info[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            pass
    try:
        import torch

        info["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        pass
    return info
