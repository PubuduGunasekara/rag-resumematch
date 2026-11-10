"""Copies an existing index into a new folder with the current HNSW search
settings (retrieval/vector_store.py, HNSW_SETTINGS), without re-embedding.

ChromaDB applies search settings only when a collection is created, so an
index built with the defaults keeps them forever. This reads every
chunk's stored embedding, text and metadata and adds them to a new
collection: the same corpus and the same vectors, a better search index.
Minutes, against about 40 for a full rebuild from the dataset.

Usage:
    python scripts/rebuild_index_settings.py chroma_db chroma_db_new
    # check it, then swap the folders yourself
"""

import argparse
import json
import time
from pathlib import Path

import chromadb
from chromadb.config import Settings

from resumematch.retrieval.vector_store import HNSW_SETTINGS

COLLECTION = "job_posting_chunks"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("target")
    parser.add_argument("--batch-size", type=int, default=5000)
    args = parser.parse_args()
    if Path(args.target).exists():
        raise SystemExit(f"{args.target} already exists; choose an empty target folder.")

    settings = Settings(anonymized_telemetry=False)
    source = chromadb.PersistentClient(path=args.source, settings=settings).get_collection(COLLECTION)
    total = source.count()
    start = time.time()
    # Read everything before opening the target: with chromadb 0.5.23, once a second
    # client has created a collection, get() on the source returns ids but no
    # embeddings (measured 2026-11-09). About 1 GB in memory for the full index.
    batches = []
    for offset in range(0, total, args.batch_size):
        batch = source.get(limit=args.batch_size, offset=offset, include=["embeddings", "documents", "metadatas"])
        if len(batch["embeddings"]) != len(batch["ids"]):
            raise SystemExit(f"Batch at {offset} came back without embeddings; nothing was written.")
        batches.append(batch)
        print(f"read {min(offset + args.batch_size, total):,} / {total:,} ({time.time() - start:.0f} s)", flush=True)

    target = chromadb.PersistentClient(path=args.target, settings=settings).create_collection(
        COLLECTION, metadata=HNSW_SETTINGS)
    written = 0
    for batch in batches:
        target.add(ids=batch["ids"], embeddings=batch["embeddings"], documents=batch["documents"],
                   metadatas=batch["metadatas"])
        written += len(batch["ids"])
        print(f"wrote {written:,} / {total:,} ({time.time() - start:.0f} s)", flush=True)

    if target.count() != total:
        raise SystemExit(f"Copied {target.count():,} of {total:,} chunks; not marking the index complete.")
    marker = Path(args.source) / "BUILD_COMPLETE.json"
    if marker.exists():
        info = json.loads(marker.read_text())
        info.update({"hnsw": HNSW_SETTINGS, "rebuilt_from": str(args.source)})
        (Path(args.target) / "BUILD_COMPLETE.json").write_text(json.dumps(info, indent=2))
    print(f"Done: {total:,} chunks in {args.target} with {HNSW_SETTINGS} in {time.time() - start:.0f} s.")


if __name__ == "__main__":
    main()
