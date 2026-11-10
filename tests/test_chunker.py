"""Tests for chunk_job_posting (sentence-aware packing, overlap,
short-tail folding, boilerplate removal)."""

from resumematch.ingest.job_postings_loader import JobPosting
from resumematch.preprocess.chunker import chunk_job_posting


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


def test_short_skills_desc_is_folded_into_description():
    posting = make_posting(skills_desc="Docker, Kubernetes, AWS")
    chunks = chunk_job_posting(posting, chunk_size=300, chunk_overlap=50)
    assert len(chunks) == 1 and chunks[0].section == "description"
    assert chunks[0].text.endswith("Skills: Docker, Kubernetes, AWS")


def test_long_skills_desc_produces_separate_section_chunks():
    posting = make_posting(skills_desc=sentences(6))  # 60 words
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


def sentences(n: int, words_each: int = 10) -> str:
    return " ".join(f"Requirement {i} " + " ".join(["detail"] * (words_each - 3)) + " here." for i in range(n))


def test_chunks_end_on_sentence_boundaries():
    posting = make_posting(description=sentences(40))  # 400 words, 10 per sentence
    chunks = chunk_job_posting(posting, chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 1
    assert all(c.text.endswith("here.") for c in chunks)
    assert all(len(c.text.split()) <= 100 for c in chunks)


def test_overlap_repeats_whole_sentences():
    chunks = chunk_job_posting(make_posting(description=sentences(40)), chunk_size=100, chunk_overlap=20)
    first_tail = chunks[0].text.split("here.")[-3].strip()  # last full sentence of chunk 0
    assert first_tail and chunks[1].text.startswith(first_tail)


def test_every_sentence_survives_including_repeated_ones():
    # Repeated sentences must not be mistaken for overlap and dropped.
    text = "Ensure compliance with all company policies at all times. " * 30 + "Final unique closing sentence with words."
    chunks = chunk_job_posting(make_posting(description=text), chunk_size=60, chunk_overlap=15)
    assert "Final unique closing sentence" in chunks[-1].text
    assert sum(c.text.count("Ensure compliance") for c in chunks) >= 30


def test_short_tail_is_folded_into_previous_chunk():
    text = sentences(10) + " Short tail."  # 100 words + 2
    chunks = chunk_job_posting(make_posting(description=text), chunk_size=100, chunk_overlap=0)
    assert len(chunks) == 1
    assert chunks[0].text.endswith("Short tail.")


def test_glued_sentences_are_split():
    # LinkedIn text often loses the space between bullets: "safely.Use"
    text = sentences(9) + " Operate the furnace safely.Use heat-treat charts daily and keep logs."
    chunks = chunk_job_posting(make_posting(description=text), chunk_size=300, chunk_overlap=0)
    # split into two sentences, re-joined with a space (unsplit would stay "safely.Use")
    assert "safely. Use heat-treat" in chunks[0].text
    assert "safely.Use" not in chunks[0].text


def test_boilerplate_and_legal_text_are_removed_but_requirements_kept():
    text = (
        "Build data pipelines in Python and SQL for analytics. "
        "We are an equal opportunity employer and value diversity at our company. "
        "This job description is subject to change at any time by the employer. "
        "Must be able to lift 50 pounds with or without reasonable accommodation."
    )
    frequent = {"this job description is subject to change at any time by the employer."}
    chunk_text = chunk_job_posting(make_posting(description=text), boilerplate=frequent)[0].text
    assert "Python and SQL" in chunk_text
    assert "equal opportunity" not in chunk_text
    assert "subject to change" not in chunk_text
    assert "lift 50 pounds" in chunk_text  # requirement exemption beats the legal pattern


def test_all_boilerplate_field_produces_no_chunks():
    posting = make_posting(description="We are an equal opportunity employer and value diversity at our company.")
    assert chunk_job_posting(posting) == []
