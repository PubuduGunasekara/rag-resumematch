"""One-time (or periodic) batch job: download the job-postings dataset,
chunk every posting, embed, and write to the persisted Chroma store.

Deliberately separate from the Streamlit app: indexing 124k+ postings
is a slow batch job that should run once (or on a schedule), not on
every app start or every user query. Run this before `streamlit run`,
not instead of it.

Usage:
    python scripts/build_index.py [--limit N]

--limit is for local dev: index only the first N postings so you can
iterate on chunking/embedding without re-processing 124k rows every
time you change something. A demo meant to show results at full
scale should use the full index, without --limit.
"""

import argparse
import dataclasses
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from resumematch.ingest.job_postings_loader import (
    download_dataset,
    iter_company_descriptions,
    iter_job_postings,
    iter_posting_texts,
    load_posting_details,
)
from resumematch.preprocess.boilerplate import find_boilerplate
from resumematch.preprocess.chunker import chunk_job_posting
from resumematch.preprocess.query_correction import build_domain_vocabulary
from resumematch.preprocess.redact import redact_contacts
from resumematch.retrieval.embedder import Embedder
from resumematch.retrieval.vector_store import VectorStore

# The query-correction vocabulary is always built from the FULL corpus,
# independent of --limit (which only controls how much gets INDEXED):
# it's one ~15s streaming pass, and a small sample makes the corrector
# "fix" real tech terms it hasn't seen (see query_correction.py).
# Written for inspection/reproducibility: exactly which sentences were
# stripped from postings as boilerplate (see preprocess/boilerplate.py).
# The vocabulary, posting details (location, pay, ... for the UI) and boilerplate list
# are written to --processed-dir, data/processed by default.
# Written into the index folder LAST, only after everything succeeded.
# entrypoint.sh checks for it, so a build that crashed halfway is
# rebuilt instead of being mistaken for a finished index.
COMPLETE_MARKER = "BUILD_COMPLETE.json"


def main():
    # Pull KAGGLE_*/EMBEDDING_MODEL/CHROMA_PERSIST_DIR from .env for local
    # runs. Real env vars (e.g. docker-compose) take precedence.
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Only index the first N postings (dev/testing only)")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"),
                        help="where the vocabulary, posting details and boilerplate list go")
    parser.add_argument("--redact-contacts", action="store_true",
                        help="replace emails and phone numbers in posting text (for an index you share)")
    args = parser.parse_args()

    persist_dir = os.environ.get("CHROMA_PERSIST_DIR", "./chroma_db")
    processed = args.processed_dir
    processed.mkdir(parents=True, exist_ok=True)
    redact = redact_contacts if args.redact_contacts else (lambda text: text)
    embedding_model = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

    print("Downloading dataset (requires KAGGLE_USERNAME/KAGGLE_KEY env vars)...")
    csv_dir = download_dataset()

    # Boilerplate is learned from the FULL (de-duplicated) corpus even with
    # --limit: "appears across >= 10 companies" is only meaningful at scale.
    print("Finding boilerplate sentences across the full corpus...")
    boilerplate = find_boilerplate(iter_company_descriptions(csv_dir))
    (processed / "boilerplate_sentences.json").write_text(json.dumps(sorted(redact(b) for b in boilerplate), indent=0))
    print(f"Found {len(boilerplate)} boilerplate sentences (written to {processed}).")

    embedder = Embedder(model_name=embedding_model)
    store = VectorStore(persist_dir=persist_dir)

    total_indexed = 0
    for batch in iter_job_postings(csv_dir, batch_size=args.batch_size):
        if args.limit is not None and total_indexed >= args.limit:
            break
        if args.limit is not None:
            batch = batch[: max(0, args.limit - total_indexed)]

        all_chunks = []
        for posting in batch:
            if args.redact_contacts:
                posting = dataclasses.replace(posting, description=redact(posting.description),
                                              skills_desc=redact(posting.skills_desc))
            all_chunks.extend(chunk_job_posting(posting, boilerplate=boilerplate))

        if not all_chunks:
            continue

        embeddings = embedder.embed([c.text for c in all_chunks])
        store.add_chunks(all_chunks, embeddings)

        total_indexed += len(batch)
        print(f"Indexed {total_indexed} postings ({store.count()} chunks total)...")

    print(f"Done. {total_indexed} postings indexed, {store.count()} chunks in {persist_dir}.")

    print("Building query-correction vocabulary from the full corpus...")
    vocabulary = build_domain_vocabulary(redact(t) for t in iter_posting_texts(csv_dir))
    (processed / "domain_vocab.json").write_text(json.dumps(vocabulary))
    print(f"Wrote {len(vocabulary)}-word domain vocabulary to {processed}.")

    details = load_posting_details(csv_dir)
    (processed / "posting_details.json").write_text(json.dumps(details))
    print(f"Wrote details for {len(details)} postings to {processed}.")

    marker = {"postings": total_indexed, "chunks": store.count(), "limit": args.limit,
              "embedding_model": embedding_model, "boilerplate_sentences": len(boilerplate),
              "contacts_redacted": args.redact_contacts}
    (Path(persist_dir) / COMPLETE_MARKER).write_text(json.dumps(marker, indent=2))
    print(f"Index complete: {marker}")


if __name__ == "__main__":
    main()
