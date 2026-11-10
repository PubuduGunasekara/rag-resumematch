"""How much of the true nearest-neighbour set does the vector index return?

For every section of every resume given, compares ChromaDB's top-k chunks
with an exact search over all stored vectors (squared L2, the collection's
space), and reports recall@k and the query time. Run it twice with
different PYTHONHASHSEED values to check that results repeat across
processes.

Usage:
    python scripts/eval_search_recall.py --index chroma_db eval/resumes/*.pdf
"""

import argparse
import hashlib
import statistics
import time
from pathlib import Path

import chromadb
import numpy as np
from chromadb.config import Settings

from resumematch.ingest.parse_resume import parse_resume
from resumematch.preprocess.resume_sections import split_resume
from resumematch.retrieval.embedder import Embedder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("resumes", nargs="+", type=Path)
    parser.add_argument("--index", default="chroma_db")
    parser.add_argument("--k", type=int, default=30)
    args = parser.parse_args()

    coll = chromadb.PersistentClient(path=args.index, settings=Settings(anonymized_telemetry=False)) \
        .get_collection("job_posting_chunks")
    print(f"Index: {args.index}, {coll.count():,} chunks, settings {coll.metadata or 'ChromaDB defaults'}")
    stored = coll.get(include=["embeddings"])
    matrix, ids = np.asarray(stored["embeddings"], dtype=np.float32), stored["ids"]

    embedder = Embedder("BAAI/bge-small-en-v1.5")
    recalls, times, fingerprint = [], [], hashlib.md5()
    for path in args.resumes:
        sections = split_resume(parse_resume(path), embedder.count_tokens, max_tokens=embedder.max_tokens - 32)
        vectors = embedder.embed([s.text for s in sections])
        coll.query(query_embeddings=vectors[:1], n_results=args.k)  # warm-up
        start = time.perf_counter()
        hits = coll.query(query_embeddings=vectors, n_results=args.k)["ids"]
        times.append((time.perf_counter() - start) / len(vectors))
        fingerprint.update(str(hits).encode())
        for vector, got in zip(vectors, hits, strict=True):
            distance = ((matrix - np.asarray(vector, dtype=np.float32)) ** 2).sum(axis=1)
            exact = {ids[i] for i in np.argpartition(distance, args.k)[: args.k]}
            recalls.append(len(exact & set(got)) / args.k)

    print(f"Resumes: {len(args.resumes)}, sections: {len(recalls)}")
    print(f"recall@{args.k} against exact search: mean {statistics.mean(recalls):.3f}, "
          f"min {min(recalls):.3f}, sections below 0.9: {sum(r < 0.9 for r in recalls)}")
    print(f"query time per section: median {statistics.median(times) * 1000:.1f} ms")
    print(f"result fingerprint (compare across PYTHONHASHSEED values): {fingerprint.hexdigest()[:10]}")


if __name__ == "__main__":
    main()
