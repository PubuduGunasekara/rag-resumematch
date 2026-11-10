"""Checks that every claim the app shows comes from the retrieved corpus.

For each resume, runs the real pipeline and verifies, for every claim
that reaches the screen:
  1. it cites one of the postings retrieved from the index for this resume;
  2. its excerpt is that posting's own text, not anything else;
  3. that posting's text contains the requirement's key terms;
  4. for a "shown" requirement, the quote is in the resume.
It also counts the model's claims that were dropped because no retrieved
posting mentions them: what the model "knew" from training, filtered out.

Needs the built index and a running Ollama.

Usage:
    python scripts/audit_grounding.py                 # eval/resumes/*.docx, 2 runs each
    python scripts/audit_grounding.py RESUME.pdf --runs 3
"""

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from resumematch.generation.gap_analysis import _mentions, evidence_in_resume, key_terms
from resumematch.generation.job_report import requirement_excerpt
from resumematch.pipeline import load_config, resolve_persist_dir, run_pipeline
from resumematch.retrieval.vector_store import VectorStore

NOT_IN_CORPUS = "none of the retrieved postings mention this requirement"


def audit(path: Path, vocab: dict | None, details: dict, store: VectorStore) -> dict:
    result = run_pipeline(str(path), domain_vocabulary=vocab, details=details)
    analysis = result.report.analysis
    retrieved = {p["job_id"] for p in result.retrieved}
    # The exact text the model was shown for each retrieved posting, rebuilt from the index.
    shown_to_model = {p["job_id"]: requirement_excerpt(store.get_posting_chunks(p["job_id"]) or [p])[0]["text"]
                      for p in result.retrieved}
    failures = []
    shown = 0
    for claim in analysis.matched + analysis.missing:
        shown += 1
        if claim.job_id not in retrieved:
            failures.append(f"cites {claim.job_id}, which was not retrieved")
        elif claim.supporting_excerpt != shown_to_model[claim.job_id]:
            failures.append(f"excerpt for '{claim.requirement}' is not that posting's text")
        if not _mentions(key_terms(claim.requirement), claim.supporting_excerpt):
            failures.append(f"'{claim.requirement}' not in its cited excerpt")
        if claim in analysis.matched and not evidence_in_resume(claim.resume_evidence or "", result.resume_text):
            failures.append(f"quote for '{claim.requirement}' not in the resume")
    excerpt_ids = {c.job_id for c in analysis.matched + analysis.missing}
    return {
        "shown": shown,
        "failures": failures,
        "cited_postings_all_retrieved": excerpt_ids <= retrieved,
        "dropped_not_in_corpus": sum(c.issue == NOT_IN_CORPUS for c in analysis.unverified),
        "dropped_total": len(analysis.unverified),
        "proposed_total": shown + len(analysis.unverified),
    }


def main():
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("resumes", nargs="*", type=Path)
    parser.add_argument("--runs", type=int, default=2)
    args = parser.parse_args()
    paths = args.resumes or sorted(Path("eval/resumes").glob("*.docx"))
    vocab_path, details_path = Path("data/processed/domain_vocab.json"), Path("data/processed/posting_details.json")
    vocab = json.loads(vocab_path.read_text()) if vocab_path.exists() else None
    details = json.loads(details_path.read_text()) if details_path.exists() else {}
    store = VectorStore(persist_dir=resolve_persist_dir(load_config()))

    print(f"{'resume':<22}{'run':>4}{'proposed':>10}{'shown':>7}{'dropped':>9}{'not in corpus':>15}{'grounding failures':>20}")
    totals = {"proposed": 0, "shown": 0, "not_in_corpus": 0, "failures": 0}
    for path in paths:
        for run in range(1, args.runs + 1):
            r = audit(path, vocab, details, store)
            print(f"{path.name[:21]:<22}{run:>4}{r['proposed_total']:>10}{r['shown']:>7}{r['dropped_total']:>9}"
                  f"{r['dropped_not_in_corpus']:>15}{len(r['failures']):>20}")
            for failure in r["failures"]:
                print(f"    FAIL: {failure}")
            totals["proposed"] += r["proposed_total"]
            totals["shown"] += r["shown"]
            totals["not_in_corpus"] += r["dropped_not_in_corpus"]
            totals["failures"] += len(r["failures"])
    print(f"\nTotal: {totals['proposed']} claims proposed by the model, {totals['shown']} shown, "
          f"{totals['not_in_corpus']} dropped because no retrieved posting mentions them, "
          f"{totals['failures']} shown claims that fail a grounding check.")


if __name__ == "__main__":
    main()
