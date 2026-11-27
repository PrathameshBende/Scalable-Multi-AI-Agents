"""Convert an LLM to GGUF with llama.cpp, quantize it, and measure size and quality.

Steps (each skipped if its output already exists, so the script can resume):
  1. clone llama.cpp and build llama-quantize + llama-perplexity (CUDA if available)
  2. download Qwen/Qwen2.5-0.5B-Instruct from Hugging Face
  3. convert_hf_to_gguf.py -> f16 GGUF (the unquantized reference)
  4. llama-quantize -> Q8_0 and Q4_K_M
  5. llama-perplexity on WikiText-2 test  (lower = better)
  6. llama-perplexity --hellaswag         (accuracy %, higher = better)

Usage (Colab): python scripts/run_gguf_eval.py
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scalable_agents.compress import file_size_mb, parse_hellaswag, parse_perplexity  # noqa: E402

HELLASWAG_URL = "https://raw.githubusercontent.com/klosax/hellaswag_text_data/main/hellaswag_val_full.txt"


def sh(cmd: list[str], log: str | None = None, cwd: str | None = None) -> str:
    print("$", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if log:
        with open(log, "w", encoding="utf-8") as f:
            f.write(p.stdout)
    if p.returncode != 0:
        print(p.stdout[-3000:])
        raise SystemExit(f"command failed ({p.returncode}): {' '.join(cmd)}")
    return p.stdout


def build_llama_cpp(root: str) -> str:
    if not os.path.isdir(root):
        sh(["git", "clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp", root])
    bin_dir = os.path.join(root, "build", "bin")
    if os.path.exists(os.path.join(bin_dir, "llama-perplexity")):
        return bin_dir
    cuda = shutil.which("nvcc") is not None or os.path.exists("/usr/local/cuda/bin/nvcc")
    flags = ["-DLLAMA_CURL=OFF"]
    if cuda:
        # sm_75 = Tesla T4; building one architecture keeps compile time down.
        flags += ["-DGGML_CUDA=ON", "-DCMAKE_CUDA_ARCHITECTURES=75",
                  "-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc"]
        # CMake needs libcuda.so (CUDA::cuda_driver). Kaggle keeps it in /usr/local/nvidia/lib64,
        # which CMake does not search, so point it at whichever of these folders has it.
        lib_dirs = [d for d in ("/usr/local/cuda/lib64/stubs", "/usr/local/nvidia/lib64",
                                "/usr/lib/x86_64-linux-gnu", "/usr/lib64")
                    if os.path.exists(os.path.join(d, "libcuda.so"))]
        if lib_dirs:
            flags.append("-DCMAKE_LIBRARY_PATH=" + ";".join(lib_dirs))
    sh(["cmake", "-B", "build", *flags], cwd=root)
    sh(["cmake", "--build", "build", "--config", "Release", "-j", str(os.cpu_count() or 2),
        "--target", "llama-quantize", "llama-perplexity"], cwd=root)
    return bin_dir


def wikitext_file(path: str) -> str:
    if not os.path.exists(path):
        from datasets import load_dataset

        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
        with open(path, "w", encoding="utf-8") as f:
            f.write("".join(ds["text"]))
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--quants", default="Q8_0,Q4_K_M")
    ap.add_argument("--workdir", default="work/gguf")
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--ppl-chunks", type=int, default=0, help="0 = whole WikiText-2 test set")
    ap.add_argument("--hellaswag-tasks", type=int, default=2000)
    ap.add_argument("--ngl", type=int, default=99, help="layers on GPU (ignored on CPU builds)")
    ap.add_argument("--out", default="results/gguf.json")
    args = ap.parse_args()

    os.makedirs(args.workdir, exist_ok=True)
    llama = os.path.abspath(os.path.join(args.workdir, "llama.cpp"))
    bin_dir = build_llama_cpp(llama)
    # Load WikiText (imports datasets/pandas/numpy) BEFORE the pip install below: that install
    # can upgrade numpy on disk, and importing datasets afterwards in this same process then
    # fails with "numpy.ufunc has no attribute __qualname__" (happened on Kaggle).
    wiki = wikitext_file(os.path.join(args.workdir, "wiki.test.raw"))
    sh([sys.executable, "-m", "pip", "install", "-q", os.path.join(llama, "gguf-py"),
        "sentencepiece", "protobuf", "mistral-common"])

    from huggingface_hub import snapshot_download

    hf_dir = snapshot_download(args.model)
    hf_size = sum(file_size_mb(os.path.join(hf_dir, f)) for f in os.listdir(hf_dir)
                  if f.endswith(".safetensors"))
    stem = args.model.split("/")[-1]
    f16 = os.path.join(args.workdir, f"{stem}-F16.gguf")
    if not os.path.exists(f16):
        sh([sys.executable, os.path.join(llama, "convert_hf_to_gguf.py"), hf_dir,
            "--outtype", "f16", "--outfile", f16], log=os.path.join(args.workdir, "convert.log"))
    files = {"F16": f16}
    for q in args.quants.split(","):
        path = os.path.join(args.workdir, f"{stem}-{q}.gguf")
        if not os.path.exists(path):
            sh([os.path.join(bin_dir, "llama-quantize"), f16, path, q],
               log=os.path.join(args.workdir, f"quantize_{q}.log"))
        files[q] = path

    hs = os.path.join(args.workdir, "hellaswag_val_full.txt")
    if not os.path.exists(hs):
        urllib.request.urlretrieve(HELLASWAG_URL, hs)

    results = {}
    for name, path in files.items():
        ppl_cmd = [os.path.join(bin_dir, "llama-perplexity"), "-m", path, "-f", wiki,
                   "-c", str(args.ctx), "-ngl", str(args.ngl)]
        if args.ppl_chunks:
            ppl_cmd += ["--chunks", str(args.ppl_chunks)]
        ppl = parse_perplexity(sh(ppl_cmd, log=os.path.join(args.workdir, f"ppl_{name}.log")))
        hs_out = sh([os.path.join(bin_dir, "llama-perplexity"), "-m", path, "-f", hs, "--hellaswag",
                     "--hellaswag-tasks", str(args.hellaswag_tasks), "-c", "512",
                     "-ngl", str(args.ngl)], log=os.path.join(args.workdir, f"hellaswag_{name}.log"))
        results[name] = {"size_mb": file_size_mb(path), **ppl, "hellaswag": parse_hellaswag(hs_out)}
        r = results[name]
        print(f"{name:>7}: {r['size_mb']:7.1f} MB  PPL {r['ppl']:.3f}  HellaSwag {r['hellaswag']['acc']:.2f}%")

    ref = results["F16"]
    for name, r in results.items():
        if name == "F16":
            continue
        r["vs_f16"] = {
            "size_reduction_pct": 100 * (1 - r["size_mb"] / ref["size_mb"]),
            "ppl_increase_pct": 100 * (r["ppl"] / ref["ppl"] - 1),
            "hellaswag_abs_points": r["hellaswag"]["acc"] - ref["hellaswag"]["acc"],
        }
        v = r["vs_f16"]
        print(f"{name} vs F16: size -{v['size_reduction_pct']:.1f}%, PPL +{v['ppl_increase_pct']:.2f}%, "
              f"HellaSwag {v['hellaswag_abs_points']:+.2f} pts")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"experiment": "gguf_llm", "model": args.model, "hf_safetensors_mb": hf_size,
                   "ctx": args.ctx, "hellaswag_tasks": args.hellaswag_tasks,
                   "llama_cpp_commit": sh(["git", "rev-parse", "HEAD"], cwd=llama).strip(),
                   "variants": results}, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
