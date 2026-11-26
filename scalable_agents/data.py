"""BEIR dataset loading (SciFact by default).

SciFact: 5,183 scientific abstracts; 300 test claims, each with expert-labelled
relevant abstracts. Public, small (~3 MB), and a standard HyDE benchmark.
Source: BEIR (Thakur et al., NeurIPS 2021 Datasets and Benchmarks).
"""

from __future__ import annotations

import csv
import gzip
import json
import os
import urllib.request
import zipfile
from dataclasses import dataclass

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"


@dataclass
class BeirData:
    corpus: dict[str, str]  # doc id -> "title text"
    queries: dict[str, str]  # query id -> text (only queries that have qrels)
    qrels: dict[str, dict[str, int]]  # query id -> {doc id: relevance}


def download_beir(name: str = "scifact", dest: str = "data") -> str:
    """Official BEIR zip; falls back to the Hugging Face mirror (BeIR/<name>) if it is down."""
    folder = os.path.join(dest, name)
    if os.path.exists(os.path.join(folder, "corpus.jsonl")):
        return folder
    os.makedirs(dest, exist_ok=True)
    zip_path = os.path.join(dest, f"{name}.zip")
    try:
        with urllib.request.urlopen(BEIR_URL.format(name=name), timeout=30) as r, \
                open(zip_path, "wb") as f:
            f.write(r.read())
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(dest)
    except Exception as e:  # server timeouts happen; the HF copy has the same files
        print(f"BEIR server failed ({e}); using Hugging Face mirror BeIR/{name}")
        _download_from_hf(name, folder)
    return folder


def _rows(path: str):
    if path.endswith(".parquet"):
        import pandas as pd

        yield from pd.read_parquet(path).to_dict("records")
    elif path.endswith(".tsv"):
        with open(path, encoding="utf-8") as f:
            yield from csv.DictReader(f, delimiter="\t")
    else:
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def _find(root: str, key: str) -> list[str]:
    exts = (".jsonl", ".jsonl.gz", ".parquet", ".tsv")
    hits = []
    for dirpath, _, files in os.walk(root):
        for fn in files:
            rel = os.path.relpath(os.path.join(dirpath, fn), root).replace("\\", "/")
            if key in rel.lower() and fn.endswith(exts):
                hits.append(os.path.join(dirpath, fn))
    return sorted(hits)


def _download_from_hf(name: str, folder: str) -> None:
    from huggingface_hub import snapshot_download

    main = snapshot_download(f"BeIR/{name}", repo_type="dataset")
    qrels = snapshot_download(f"BeIR/{name}-qrels", repo_type="dataset")
    os.makedirs(os.path.join(folder, "qrels"), exist_ok=True)
    for key in ("queries", "corpus"):  # corpus last: its presence marks success
        files = _find(main, key)
        if not files:
            raise FileNotFoundError(f"no {key} file in BeIR/{name}")
        with open(os.path.join(folder, f"{key}.jsonl"), "w", encoding="utf-8") as out:
            for path in files:
                for row in _rows(path):
                    out.write(json.dumps({"_id": str(row["_id"]), "title": row.get("title", ""),
                                          "text": row.get("text", "")}) + "\n")
        if key == "queries":
            for split in ("train", "dev", "test"):
                rows = [r for p in _find(qrels, split) for r in _rows(p)]
                if rows:
                    with open(os.path.join(folder, "qrels", f"{split}.tsv"), "w",
                              encoding="utf-8", newline="") as f:
                        w = csv.writer(f, delimiter="\t")
                        w.writerow(["query-id", "corpus-id", "score"])
                        for r in rows:
                            w.writerow([r["query-id"], r["corpus-id"], int(r["score"])])


def load_beir(folder: str, split: str = "test") -> BeirData:
    corpus: dict[str, str] = {}
    with open(os.path.join(folder, "corpus.jsonl"), encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            corpus[str(row["_id"])] = f"{row.get('title', '')} {row.get('text', '')}".strip()

    qrels: dict[str, dict[str, int]] = {}
    with open(os.path.join(folder, "qrels", f"{split}.tsv"), encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # header: query-id corpus-id score
        for qid, did, score in reader:
            qrels.setdefault(str(qid), {})[str(did)] = int(score)

    queries: dict[str, str] = {}
    with open(os.path.join(folder, "queries.jsonl"), encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if str(row["_id"]) in qrels:
                queries[str(row["_id"])] = row["text"]
    return BeirData(corpus=corpus, queries=queries, qrels=qrels)
