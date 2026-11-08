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
time you change something. Never use --limit for the actual GCP/
presentation-day index — that one needs the real dataset scale the
rubric asks for.
"""

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from resumematch.ingest.job_postings_loader import (
    download_dataset,
    iter_company_descriptions,
    iter_job_postings,
    iter_posting_texts,
)
from resumematch.preprocess.boilerplate import find_boilerplate
from resumematch.preprocess.chunker import chunk_job_posting
from resumematch.preprocess.query_correction import build_domain_vocabulary
from resumematch.retrieval.embedder import Embedder
from resumematch.retrieval.vector_store import VectorStore

# The query-correction vocabulary is always built from the FULL corpus,
# independent of --limit (which only controls how much gets INDEXED):
# it's one ~15s streaming pass, and a small sample makes the corrector
# "fix" real tech terms it hasn't seen (see query_correction.py).
VOCAB_OUTPUT_PATH = Path("data/processed/domain_vocab.json")
# Written for inspection/reproducibility: exactly which sentences were
# stripped from postings as boilerplate (see preprocess/boilerplate.py).
BOILERPLATE_OUTPUT_PATH = Path("data/processed/boilerplate_sentences.json")


def main():
    # Pull KAGGLE_*/EMBEDDING_MODEL/CHROMA_PERSIST_DIR from .env for local
    # runs. Real env vars (e.g. docker-compose) take precedence.
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Only index the first N postings (dev/testing only)")
    parser.add_argument("--batch-size", type=int, default=500)
    args = parser.parse_args()

    persist_dir = os.environ.get("CHROMA_PERSIST_DIR", "./chroma_db")
    embedding_model = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

    print("Downloading dataset (requires KAGGLE_USERNAME/KAGGLE_KEY env vars)...")
    csv_dir = download_dataset()

    # Boilerplate is learned from the FULL (de-duplicated) corpus even with
    # --limit: "appears across >= 10 companies" is only meaningful at scale.
    print("Finding boilerplate sentences across the full corpus...")
    boilerplate = find_boilerplate(iter_company_descriptions(csv_dir))
    BOILERPLATE_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    BOILERPLATE_OUTPUT_PATH.write_text(json.dumps(sorted(boilerplate), indent=0))
    print(f"Found {len(boilerplate)} boilerplate sentences (written to {BOILERPLATE_OUTPUT_PATH}).")

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
            all_chunks.extend(chunk_job_posting(posting, boilerplate=boilerplate))

        if not all_chunks:
            continue

        embeddings = embedder.embed([c.text for c in all_chunks])
        store.add_chunks(all_chunks, embeddings)

        total_indexed += len(batch)
        print(f"Indexed {total_indexed} postings ({store.count()} chunks total)...")

    print(f"Done. {total_indexed} postings indexed, {store.count()} chunks in {persist_dir}.")

    print("Building query-correction vocabulary from the full corpus...")
    vocabulary = build_domain_vocabulary(iter_posting_texts(csv_dir))
    VOCAB_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    VOCAB_OUTPUT_PATH.write_text(json.dumps(vocabulary))
    print(f"Wrote {len(vocabulary)}-word domain vocabulary to {VOCAB_OUTPUT_PATH}.")


if __name__ == "__main__":
    main()
