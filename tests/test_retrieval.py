"""Tests for VectorStore. Uses fake fixed-length vectors to stay fast and
deterministic — real embeddings aren't needed to verify the store's
add/query contract, just its wiring.
"""

import pytest
import tempfile
from resumematch.retrieval.vector_store import VectorStore
from resumematch.preprocess.chunker import Chunk


@pytest.fixture
def store():
    with tempfile.TemporaryDirectory() as tmp:
        yield VectorStore(persist_dir=tmp, collection_name="test_chunks")


def make_chunk(text, job_id="1", section="description", url="https://example.com/1", title="Engineer"):
    return Chunk(text=text, chunk_id=f"{job_id}_{section}_0", job_id=job_id, section=section,
                 job_posting_url=url, title=title)


def test_add_then_query_returns_added_chunk(store):
    chunks = [make_chunk("Experience with Python and FastAPI.")]
    embeddings = [[1.0, 0.0, 0.0]]
    store.add_chunks(chunks, embeddings)

    results = store.query([1.0, 0.0, 0.0], top_k=1)
    assert len(results) == 1
    assert "Python" in results[0]["text"]
    assert results[0]["job_id"] == "1"
    assert results[0]["job_posting_url"] == "https://example.com/1"


def test_query_returns_most_similar_first(store):
    chunks = [
        make_chunk("Backend engineer, Python, Django.", job_id="1"),
        make_chunk("Painted houses for three summers.", job_id="2"),
    ]
    embeddings = [[1.0, 0.0], [0.0, 1.0]]
    store.add_chunks(chunks, embeddings)

    results = store.query([0.9, 0.1], top_k=2)
    assert results[0]["job_id"] == "1"


def test_count_reflects_added_chunks(store):
    chunks = [make_chunk("A", job_id="1"), make_chunk("B", job_id="2")]
    embeddings = [[1.0, 0.0], [0.0, 1.0]]
    assert store.count() == 0
    store.add_chunks(chunks, embeddings)
    assert store.count() == 2
