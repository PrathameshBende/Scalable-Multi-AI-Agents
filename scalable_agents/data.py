"""BEIR dataset loading (SciFact by default).

SciFact: 5,183 scientific abstracts; 300 test claims, each with expert-labelled
relevant abstracts. Public, small (~3 MB), and a standard HyDE benchmark.
Source: BEIR (Thakur et al., NeurIPS 2021 Datasets and Benchmarks).
"""

from __future__ import annotations

import csv
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
    folder = os.path.join(dest, name)
    if os.path.isdir(folder):
        return folder
    os.makedirs(dest, exist_ok=True)
    zip_path = os.path.join(dest, f"{name}.zip")
    urllib.request.urlretrieve(BEIR_URL.format(name=name), zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    return folder


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
