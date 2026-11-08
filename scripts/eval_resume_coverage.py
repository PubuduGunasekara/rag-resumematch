"""Before/after for sectioned retrieval: does the whole resume count now?

Two checks, single-vector (old: whole resume as one query, silently
truncated at the embedder's 512 tokens) vs sectioned (new: one query per
section + reciprocal rank fusion):

  1. Coverage: how many of the resume's tokens each method can see, and
     the top-k posting titles each retrieves.
  2. Sensitivity: replace the resume's LAST section (typically skills)
     with an unrelated profession's skills. If a method can't see that
     section, its results won't change at all (overlap 1.00) -- which is
     exactly what truncation looks like from the outside.

Usage:
    python scripts/eval_resume_coverage.py RESUME.pdf [--top-k 10]
"""

import argparse
import os

from dotenv import load_dotenv

from resumematch.ingest.parse_resume import parse_resume
from resumematch.pipeline import load_config
from resumematch.preprocess.resume_sections import split_resume
from resumematch.retrieval.embedder import Embedder
from resumematch.retrieval.fusion import fuse_rankings, retrieve_for_resume
from resumematch.retrieval.vector_store import VectorStore

UNRELATED_SKILLS = """TECHNICAL SKILLS
Clinical: patient assessment, IV therapy, wound care, medication administration, triage
Certifications: Registered Nurse (RN), BLS, ACLS, PALS
Systems: Epic EHR, Cerner, Pyxis MedStation
Specialties: ICU, emergency department, telemetry, pediatric care, patient education"""


def main():
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("resume")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    config = load_config()
    embedder = Embedder(os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"))
    store = VectorStore(persist_dir=config.get("chroma_persist_dir", "./chroma_db"))
    per_section_k = config["retrieval"]["per_section_k"]
    budget = embedder.max_tokens - 32

    def single_vector(text: str) -> list[dict]:
        # Old behaviour, but deduplicated to postings so both methods
        # are compared on the same unit.
        hits = store.query(embedder.embed([text])[0], top_k=per_section_k)
        return fuse_rankings([("WHOLE RESUME", hits)], top_k=args.top_k, max_per_company=None)

    def sectioned(text: str) -> list[dict]:
        sections = split_resume(text, embedder.count_tokens, max_tokens=budget)
        return retrieve_for_resume(
            sections, embedder, store, top_k=args.top_k, per_section_k=per_section_k,
            max_per_company=config["retrieval"]["max_per_company"],
        )

    def overlap(a: list[dict], b: list[dict]) -> float:
        return len({r["job_id"] for r in a} & {r["job_id"] for r in b}) / max(len(a), 1)

    text = parse_resume(args.resume)
    sections = split_resume(text, embedder.count_tokens, max_tokens=budget)
    total = embedder.count_tokens(text)
    print(f"Index: {store.count()} chunks | resume: {total} tokens, embedder window: {embedder.max_tokens}")
    print(f"  single-vector sees: {min(total, embedder.max_tokens - 2)} tokens "
          f"({min(1.0, (embedder.max_tokens - 2) / total):.0%})")
    print(f"  sectioned sees:     {total} tokens (100%) as {len(sections)} sections: "
          f"{', '.join(s.heading for s in sections)}\n")

    old, new = single_vector(text), sectioned(text)
    print(f"Top {args.top_k} postings (overlap between methods: {overlap(old, new):.2f})")
    print(f"  {'single-vector (old)':<45} | sectioned + RRF (new)")
    for a, b in zip(old, new):
        print(f"  {a['title'][:45]:<45} | {b['title'][:40]}  [{', '.join(b['matched_sections'][:3])}]")

    # Sensitivity: swap the last section for an unrelated profession's.
    last = sections[-1]
    body_start = text.index(last.text.splitlines()[1]) - len(last.text.splitlines()[0]) - 1
    swapped = text[:body_start] + "\n" + UNRELATED_SKILLS
    print(f"\nSensitivity: replaced '{last.heading}' with nursing skills")
    print(f"  single-vector: top-{args.top_k} overlap with original = {overlap(old, single_vector(swapped)):.2f}"
          f"   (1.00 = the section was never seen)")
    print(f"  sectioned:     top-{args.top_k} overlap with original = {overlap(new, sectioned(swapped)):.2f}")


if __name__ == "__main__":
    main()
