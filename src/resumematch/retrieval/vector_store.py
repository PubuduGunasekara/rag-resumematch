"""Thin wrapper around chromadb. Isolates the vector-store library to
one file so swapping it later (e.g. if 100k+ rows makes a different
backend worth it) means changing one file, not grepping the project.
"""

from resumematch.preprocess.chunker import Chunk

# HNSW search settings, applied when a collection is created (ChromaDB
# ignores changes to them afterwards). With the defaults (search_ef 10),
# search returned 77% of the true 30 nearest chunks on one resume and
# different neighbours in different processes; see docs/engineering-log.md,
# search-recall. "l2" is ChromaDB's default space, kept so distances match
# indexes built before this change.
HNSW_SETTINGS = {"hnsw:space": "l2", "hnsw:construction_ef": 200, "hnsw:search_ef": 200, "hnsw:M": 16}


class VectorStore:
    def __init__(self, persist_dir: str = "./chroma_db", collection_name: str = "job_posting_chunks"):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self._client = None
        self._collection = None

    def _get_collection(self):
        if self._collection is None:
            import chromadb
            from chromadb.config import Settings
            # ChromaDB sends anonymized usage events by default (no document
            # text). Off: nothing about a session should leave the machine.
            self._client = chromadb.PersistentClient(
                path=self.persist_dir, settings=Settings(anonymized_telemetry=False)
            )
            try:
                self._collection = self._client.get_collection(self.collection_name)
            except Exception:  # noqa: BLE001 - chromadb raises different types across versions
                # Settings only take effect when a collection is created.
                self._collection = self._client.create_collection(self.collection_name, metadata=HNSW_SETTINGS)
        return self._collection

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """
        Stores chunks with their embeddings and citation metadata
        (job_id, section, job_posting_url, title): every field needed
        to render a clickable citation lives in metadata, not somewhere
        that has to be re-joined at query time.

        Called in batches from scripts/build_index.py, NOT with the
        full 124k-row corpus at once; batch size matches
        job_postings_loader's iter_job_postings batch_size.
        """
        if not chunks:
            return
        collection = self._get_collection()
        collection.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[{
                "job_id": c.job_id, "section": c.section,
                "job_posting_url": c.job_posting_url or "",
                "title": c.title,
                "company_name": c.company_name or "",
            } for c in chunks],
        )

    def query(self, query_embedding: list[float], top_k: int = 5, job_ids: list[str] | None = None) -> list[dict]:
        """
        Output: list of dicts, ordered by relevance (closest first), each:
            {"text": str, "distance": float, "job_id": str, "section": str,
             "job_posting_url": str | None, "title": str, "company_name": str | None}
        This is the shape the Streamlit UI and gap_analysis both consume
        directly, no second lookup needed to render a citation.
        """
        collection = self._get_collection()
        # job_ids: search only these postings (search filters; see retrieval/filters.py).
        where = {"job_id": {"$in": [str(j) for j in job_ids]}} if job_ids is not None else None
        result = collection.query(query_embeddings=[query_embedding], n_results=top_k, where=where)

        documents = result["documents"][0]
        distances = result["distances"][0]
        metadatas = result["metadatas"][0]

        out = []
        for text, distance, meta in zip(documents, distances, metadatas, strict=False):
            out.append({
                "text": text,
                "distance": distance,
                "job_id": meta["job_id"],
                "section": meta["section"],
                "job_posting_url": meta["job_posting_url"] or None,
                "title": meta["title"],
                "company_name": meta.get("company_name") or None,
            })
        return out

    def get_posting_chunks(self, job_id: str) -> list[dict]:
        """Every chunk of one posting, in document order (description
        before skills_desc, then by chunk index). Retrieval returns only
        a posting's best-matching chunk; a per-job analysis needs all of
        it, or requirements in the other chunks are invisible."""
        result = self._get_collection().get(where={"job_id": str(job_id)})
        rows = sorted(zip(result["ids"], result["documents"], result["metadatas"], strict=False),
                      key=lambda r: (r[2]["section"] != "description", int(r[0].rsplit("_", 1)[-1])))
        return [{
            "text": text, "distance": None, "job_id": meta["job_id"], "section": meta["section"],
            "job_posting_url": meta["job_posting_url"] or None, "title": meta["title"],
            "company_name": meta.get("company_name") or None,
        } for _, text, meta in rows]

    def iter_metadata(self, batch_size: int = 50_000):
        """Every chunk's metadata, in batches. Used once to build the
        posting-title cache for role reports (about 15 s on the full index)."""
        collection, offset = self._get_collection(), 0
        while True:
            result = collection.get(include=["metadatas"], limit=batch_size, offset=offset)
            if not result["ids"]:
                return
            yield from result["metadatas"]
            offset += batch_size

    def get_postings_text(self, job_ids: list[str], batch_size: int = 200) -> dict[str, list[dict]]:
        """job_id -> that posting's chunks ({"text", "section", ...}), for many
        postings at once. Order within a posting follows get_posting_chunks."""
        out: dict[str, list[dict]] = {}
        collection = self._get_collection()
        for start in range(0, len(job_ids), batch_size):
            batch = [str(j) for j in job_ids[start:start + batch_size]]
            result = collection.get(where={"job_id": {"$in": batch}}, include=["documents", "metadatas"])
            for chunk_id, text, meta in zip(result["ids"], result["documents"], result["metadatas"], strict=False):
                out.setdefault(meta["job_id"], []).append(
                    {"id": chunk_id, "text": text, "section": meta["section"], "title": meta["title"],
                     "job_id": meta["job_id"], "job_posting_url": meta["job_posting_url"] or None,
                     "company_name": meta.get("company_name") or None})
        for chunks in out.values():
            chunks.sort(key=lambda c: (c["section"] != "description", int(c["id"].rsplit("_", 1)[-1])))
        return out

    def count(self) -> int:
        """Number of chunks currently indexed, used by build_index.py to
        report progress and by the Streamlit app to show a "index not
        built yet" message instead of silently returning nothing."""
        return self._get_collection().count()
