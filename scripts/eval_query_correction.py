"""Does query correction actually help retrieval? A controlled experiment.

Method (synthetic noise, since we have no corpus of real OCR-damaged
resumes with clean ground truth):
  1. Retrieve top-k postings for the CLEAN resume -> the reference set.
  2. Inject typos into a fraction of eligible words (lowercase, >= 5
     letters, known to the vocabulary), one random edit each, using the
     four standard edit operations (delete/insert/replace/swap).
     Each typo is guaranteed to be a non-word, i.e. something a
     corrector could in principle detect.
  3. Retrieve for the NOISY resume, with correction OFF and ON.
  4. Report overlap@k with the clean reference (1.0 = typos had no
     effect on retrieval) plus the corrector's own precision/recall.

Also reports false corrections on the clean resume (should be 0).

Usage:
    python scripts/eval_query_correction.py RESUME.pdf [RESUME2.docx ...]
        [--rates 0.05 0.1 0.2] [--seeds 5] [--top-k 10]

Requires a built index (scripts/build_index.py) and its vocabulary.
"""

import argparse
import json
import os
import random
import re
import statistics
from pathlib import Path

from dotenv import load_dotenv

from resumematch.ingest.parse_resume import parse_resume
from resumematch.pipeline import load_config, resolve_persist_dir
from resumematch.preprocess.query_correction import LETTERS, correct_query
from resumematch.preprocess.resume_sections import split_resume
from resumematch.retrieval.embedder import Embedder
from resumematch.retrieval.fusion import retrieve_for_resume
from resumematch.retrieval.vector_store import VectorStore

VOCAB_PATH = Path("data/processed/domain_vocab.json")


def make_typo(word: str, vocab: dict[str, float], rng: random.Random) -> str:
    """One random single-character edit that yields a non-word."""
    for _ in range(50):
        i = rng.randrange(len(word))
        op = rng.choice(("delete", "insert", "replace", "swap"))
        if op == "delete":
            typo = word[:i] + word[i + 1:]
        elif op == "insert":
            typo = word[:i] + rng.choice(LETTERS) + word[i:]
        elif op == "replace":
            typo = word[:i] + rng.choice(LETTERS.replace(word[i], "")) + word[i + 1:]
        else:
            if i == len(word) - 1 or word[i] == word[i + 1]:
                continue
            typo = word[:i] + word[i + 1] + word[i] + word[i + 2:]
        if typo not in vocab:
            return typo
    return word  # couldn't make a non-word typo; leave it


def inject_typos(text: str, rate: float, vocab: dict[str, float], rng: random.Random):
    """Returns (noisy_text, {typo: original}) for the words changed."""
    injected: dict[str, str] = {}

    def maybe_corrupt(match: re.Match) -> str:
        word = match.group(0)
        eligible = word.islower() and len(word) >= 5 and word in vocab
        if not eligible or rng.random() >= rate:
            return word
        typo = make_typo(word, vocab, rng)
        if typo != word:
            injected[typo] = word
        return typo

    return re.sub(r"[A-Za-z]+", maybe_corrupt, text), injected


def overlap(a: list[str], b: list[str]) -> float:
    return len(set(a) & set(b)) / len(a)


def main():
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("resumes", nargs="+")
    parser.add_argument("--rates", type=float, nargs="+", default=[0.05, 0.1, 0.2])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    config = load_config()
    rules = config["query_correction"]
    vocab = json.loads(VOCAB_PATH.read_text())
    embedder = Embedder(os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"))
    store = VectorStore(persist_dir=resolve_persist_dir(config))
    print(f"Index: {store.count()} chunks | vocabulary: {len(vocab)} words | top-k={args.top_k}\n")

    def retrieve(text: str) -> list[str]:
        # Same retrieval as the pipeline: per-section queries + RRF, so
        # typos anywhere in the resume can matter (not just the first
        # 512 tokens). Compared on distinct postings.
        sections = split_resume(text, embedder.count_tokens, max_tokens=embedder.max_tokens - 32)
        hits = retrieve_for_resume(
            sections, embedder, store, top_k=args.top_k, per_section_k=config["retrieval"]["per_section_k"]
        )
        return [h["job_id"] for h in hits]

    for path in args.resumes:
        clean = parse_resume(path)
        _, clean_results = correct_query(clean, vocab, **rules)
        false_on_clean = [(r.original_token, r.corrected_token) for r in clean_results if r.was_corrected]
        reference = retrieve(clean)
        print(f"== {Path(path).name}")
        print(f"   false corrections on the clean resume: {len(false_on_clean)} {false_on_clean or ''}")
        print(f"   {'typo rate':>9} {'typos':>6} {'overlap OFF':>12} {'overlap ON':>11} "
              f"{'fixed exactly':>14} {'wrong fixes':>12}")

        for rate in args.rates:
            off, on, n_typos, recall, wrong = [], [], [], [], []
            for seed in range(args.seeds):
                rng = random.Random(seed)
                noisy, injected = inject_typos(clean, rate, vocab, rng)
                corrected, results = correct_query(noisy, vocab, **rules)
                fixes = [r for r in results if r.was_corrected]
                exact = sum(1 for r in fixes if injected.get(r.original_token) == r.corrected_token)
                off.append(overlap(reference, retrieve(noisy)))
                on.append(overlap(reference, retrieve(corrected)))
                n_typos.append(len(injected))
                recall.append(exact / len(injected) if injected else 1.0)
                wrong.append(len(fixes) - exact)
            m = statistics.mean
            print(f"   {rate:>9.0%} {m(n_typos):>6.0f} {m(off):>12.2f} {m(on):>11.2f} "
                  f"{m(recall):>14.0%} {m(wrong):>12.1f}")
        print()

    print("overlap = share of the clean resume's top-k postings still retrieved (1.00 = typos had no effect).")
    print("fixed exactly = injected typos restored to the original word; wrong fixes = changes that "
          "were not an exact restoration.")


if __name__ == "__main__":
    main()
