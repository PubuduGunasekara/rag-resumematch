"""Times each stage of the pipeline on real resumes and prints a table.

Runs the same steps as pipeline.run_pipeline, but calls each one
separately so it can be timed: validate and parse, typo correction,
section split, embedding, search and fusion, requirement excerpts, model
generation, and claim checks. Needs the built index and a running Ollama.

Usage:
    python scripts/benchmark.py                      # eval/resumes/*.docx, 3 runs each
    python scripts/benchmark.py RESUME.pdf --runs 5 --json out.json

The first run of each resume is a warm-up and is not counted (it loads
the embedding model and the LLM into memory). Report the hardware line
with any number you quote.
"""

import argparse
import json
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

from resumematch.generation.gap_analysis import build_prompt, parse_response, response_schema
from resumematch.generation.job_report import requirement_excerpt
from resumematch.generation.ollama_client import OllamaClient
from resumematch.ingest.validate import validate_resume
from resumematch.pipeline import load_config, resolve_persist_dir
from resumematch.preprocess.query_correction import correct_query
from resumematch.preprocess.resume_sections import split_resume
from resumematch.retrieval.embedder import Embedder
from resumematch.retrieval.fusion import fuse_rankings
from resumematch.retrieval.vector_store import VectorStore

STAGES = ["validate+parse", "correct", "split", "embed", "search", "excerpts", "generate", "check"]


def hardware() -> dict:
    cpu = platform.processor() or platform.machine()
    ram = None
    if platform.system() == "Darwin":
        cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
        ram = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout) / 2**30
    elif Path("/proc/meminfo").exists():  # Linux, including inside Docker
        kb = next(int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines()
                  if line.startswith("MemTotal"))
        ram = kb / 2**20
        cpu = f"{cpu}, {os.cpu_count()} CPUs"
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    try:
        ollama = requests.get(f"{host}/api/version", timeout=2).json().get("version")
    except requests.RequestException:
        ollama = None
    return {"cpu": cpu, "ram_gb": round(ram, 1) if ram else None, "os": f"{platform.system()} {platform.release()}",
            "python": platform.python_version(), "ollama": ollama, "ollama_host": host}


def run_once(path: Path, cfg: dict, vocab: dict | None, embedder: Embedder, store: VectorStore,
             client: OllamaClient) -> tuple[dict, dict]:
    t, stats = {}, {}

    def timed(name, fn):
        start = time.perf_counter()
        out = fn()
        t[name] = time.perf_counter() - start
        return out

    text, _ = timed("validate+parse", lambda: validate_resume(path))
    query = timed("correct", lambda: correct_query(text, vocab, **cfg["query_correction"])[0] if vocab else text)
    sections = timed("split", lambda: split_resume(query, embedder.count_tokens, max_tokens=embedder.max_tokens - 32))
    vectors = timed("embed", lambda: embedder.embed([s.text for s in sections]))
    r = cfg["retrieval"]
    retrieved = timed("search", lambda: fuse_rankings(
        [(s.heading, store.query(v, top_k=r["per_section_k"])) for s, v in zip(sections, vectors, strict=False)],
        top_k=r["top_k"], max_per_company=r["max_per_company"]))
    excerpts = timed("excerpts", lambda: [requirement_excerpt(store.get_posting_chunks(p["job_id"]) or [p])[0]
                                          for p in retrieved])
    gen, rep = cfg["generation"], cfg.get("report", {})
    prompt = build_prompt(text, excerpts, cover_each=rep.get("cover_each", False))
    schema = response_schema(len(excerpts), max_items=rep.get("max_items", 8))
    raw = timed("generate", lambda: client.generate(prompt, temperature=gen["temperature"],
                                                   max_tokens=rep.get("max_tokens", gen["max_tokens"]),
                                                   num_ctx=gen["num_ctx"], schema=schema))
    result = timed("check", lambda: parse_response(raw, excerpts, resume_text=text))
    stats = {"sections": len(sections), "claims": len(result.matched) + len(result.missing) + len(result.unverified),
             "kept": len(result.matched) + len(result.missing)}
    return t, stats


def main():
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("resumes", nargs="*", type=Path)
    parser.add_argument("--runs", type=int, default=3, help="timed runs per resume, after one warm-up")
    parser.add_argument("--json", type=Path, help="also write raw timings here")
    args = parser.parse_args()
    paths = args.resumes or sorted(Path("eval/resumes").glob("*.docx"))

    cfg = load_config()
    vocab_path = Path("data/processed/domain_vocab.json")
    vocab = json.loads(vocab_path.read_text()) if vocab_path.exists() else None
    embedder = Embedder(os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"))
    store = VectorStore(persist_dir=resolve_persist_dir(cfg))
    client = OllamaClient()
    hw = hardware()
    print(f"Hardware: {hw['cpu']}, {hw['ram_gb']} GB RAM, {hw['os']}, Python {hw['python']}, "
          f"Ollama {hw['ollama']} at {hw['ollama_host']}, model {client.model}")
    print(f"Index: {store.count():,} chunks. Runs per resume: {args.runs} (after 1 warm-up). Times in seconds.\n")

    rows = {}
    for path in paths:
        run_once(path, cfg, vocab, embedder, store, client)  # warm-up
        runs = [run_once(path, cfg, vocab, embedder, store, client) for _ in range(args.runs)]
        rows[path.name] = runs

    header = f"{'resume':<18}" + "".join(f"{s:>15}" for s in STAGES) + f"{'total':>9}"
    print(header)
    all_totals = []
    for name, runs in rows.items():
        med = {s: statistics.median(r[0][s] for r in runs) for s in STAGES}
        totals = [sum(r[0].values()) for r in runs]
        all_totals += totals
        print(f"{name:<18}" + "".join(f"{med[s]:>15.3f}" for s in STAGES) + f"{statistics.median(totals):>9.1f}")
    print(f"\nMedian total over all runs: {statistics.median(all_totals):.1f} s "
          f"(min {min(all_totals):.1f}, max {max(all_totals):.1f}, n = {len(all_totals)})")
    if args.json:
        args.json.write_text(json.dumps({"hardware": hw, "model": client.model, "runs": {
            n: [{"times": t, "stats": s} for t, s in runs] for n, runs in rows.items()}}, indent=2))


if __name__ == "__main__":
    main()
