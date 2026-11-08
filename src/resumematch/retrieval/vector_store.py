"""Thin wrapper around chromadb. Isolates the vector-store library to
one file so swapping it later (e.g. if 100k+ rows makes a different
backend worth it) means changing one file, not grepping the project.
"""

from resumematch.preprocess.chunker import Chunk


class VectorStore:
    def __init__(self, persist_dir: str = "./chroma_db", collection_name: str = "job_posting_chunks"):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self._client = None
        self._collection = None

    def _get_collection(self):
        if self._collection is None:
            import chromadb
            self._client = chromadb.PersistentClient(path=self.persist_dir)
            self._collection = self._client.get_or_create_collection(self.collection_name)
        return self._collection

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """
        Stores chunks with their embeddings and citation metadata
        (job_id, section, job_posting_url, title) — every field needed
        to render a clickable citation lives in metadata, not somewhere
        that has to be re-joined at query time.

        Called in batches from scripts/build_index.py, NOT with the
        full 124k-row corpus at once — batch size matches
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

    def query(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        """
        Output: list of dicts, ordered by relevance (closest first), each:
            {"text": str, "distance": float, "job_id": str, "section": str,
             "job_posting_url": str | None, "title": str, "company_name": str | None}
        This is the shape the Streamlit UI and gap_analysis both consume
        directly — no second lookup needed to render a citation.
        """
        collection = self._get_collection()
        result = collection.query(query_embeddings=[query_embedding], n_results=top_k)

        documents = result["documents"][0]
        distances = result["distances"][0]
        metadatas = result["metadatas"][0]

        out = []
        for text, distance, meta in zip(documents, distances, metadatas):
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

    def count(self) -> int:
        """Number of chunks currently indexed — used by build_index.py to
        report progress and by the Streamlit app to show a "index not
        built yet" message instead of silently returning nothing."""
        return self._get_collection().count()
