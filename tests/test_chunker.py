"""Tests for chunk_job_posting.

These will fail until _chunk_field is implemented — that's intentional:
write the implementation to make these pass, don't adjust the tests to
match whatever the implementation happens to produce.
"""

import pytest
from resumematch.preprocess.chunker import chunk_job_posting
from resumematch.ingest.job_postings_loader import JobPosting


def make_posting(description="Software Engineer role requiring Python and SQL.", skills_desc=None):
    return JobPosting(
        job_id="12345",
        title="Software Engineer",
        description=description,
        skills_desc=skills_desc,
        company_id="999",
        job_posting_url="https://www.linkedin.com/jobs/view/12345",
    )


def test_short_description_produces_single_chunk():
    posting = make_posting()
    chunks = chunk_job_posting(posting, chunk_size=300, chunk_overlap=50)
    assert len(chunks) == 1
    assert "Python" in chunks[0].text


def test_every_chunk_carries_citation_metadata():
    posting = make_posting()
    chunks = chunk_job_posting(posting, chunk_size=300, chunk_overlap=50)
    for c in chunks:
        assert c.job_id == "12345"
        assert c.job_posting_url == "https://www.linkedin.com/jobs/view/12345"
        assert c.section == "description"


def test_skills_desc_produces_separate_section_chunks():
    posting = make_posting(skills_desc="Required: Docker, Kubernetes, AWS.")
    chunks = chunk_job_posting(posting, chunk_size=300, chunk_overlap=50)
    sections = {c.section for c in chunks}
    assert sections == {"description", "skills_desc"}


def test_missing_skills_desc_produces_no_skills_section():
    posting = make_posting(skills_desc=None)
    chunks = chunk_job_posting(posting, chunk_size=300, chunk_overlap=50)
    assert all(c.section == "description" for c in chunks)


def test_no_chunk_is_empty():
    long_description = "Did impactful engineering work. " * 200
    posting = make_posting(description=long_description)
    chunks = chunk_job_posting(posting, chunk_size=50, chunk_overlap=10)
    assert all(c.text.strip() for c in chunks)


def test_chunk_ids_are_unique():
    long_description = "Did impactful engineering work. " * 200
    posting = make_posting(description=long_description)
    chunks = chunk_job_posting(posting, chunk_size=50, chunk_overlap=10)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
