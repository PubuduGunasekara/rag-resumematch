"""Tests for resume section splitting. count_tokens is a whitespace word
count here so the tests don't need the embedding model; the pipeline
passes the real tokenizer (Embedder.count_tokens)."""

from resumematch.preprocess.resume_sections import split_resume


def words(text: str) -> int:
    return len(text.split())


def bullets(prefix: str, n: int) -> str:
    return "\n".join(f"{prefix} bullet {i} with several words of detail" for i in range(n))


RESUME = f"""JANE DOE
jane@example.com | Boston, MA | Seeking a software engineering internship role
EDUCATION
Northeastern University, MS Computer Science, 2026 - 2028, coursework in NLP
EXPERIENCE
{bullets("Built a Kafka pipeline", 3)}
Technical Skills:
Python, Java, Kafka, Redis, PostgreSQL, Docker, Kubernetes, AWS, React, TypeScript
"""


def test_splits_on_caps_and_known_headings():
    sections = split_resume(RESUME, words, max_tokens=200)
    assert [s.heading for s in sections] == ["HEADER", "EDUCATION", "EXPERIENCE", "TECHNICAL SKILLS"]


def test_name_line_in_capitals_is_not_a_heading():
    sections = split_resume(RESUME, words, max_tokens=200)
    assert sections[0].heading == "HEADER"
    assert sections[0].text.startswith("JANE DOE")


def test_every_section_fits_the_budget_and_no_text_is_lost():
    long_resume = RESUME + "PROJECTS\n" + bullets("Shipped a service", 40)
    sections = split_resume(long_resume, words, max_tokens=60)
    assert all(words(s.text) <= 60 for s in sections)
    original_lines = {line.strip() for line in long_resume.splitlines() if line.strip()}
    covered = {line for s in sections for line in s.text.splitlines()}
    assert original_lines <= covered  # nothing dropped (continuations may repeat a heading)


def test_oversized_section_splits_evenly_with_heading_repeated():
    text = RESUME + "PROJECTS\n" + bullets("Shipped a service", 12)  # ~108 words
    sections = split_resume(text, words, max_tokens=80)
    parts = [s for s in sections if s.heading.startswith("PROJECTS")]
    assert [s.heading for s in parts] == ["PROJECTS (1/2)", "PROJECTS (2/2)"]
    assert parts[1].text.startswith("PROJECTS\n")
    sizes = [words(s.text) for s in parts]
    assert max(sizes) - min(sizes) <= 20  # balanced, not 80 + scrap


def test_tiny_section_is_merged_forward():
    text = RESUME.replace("EDUCATION\n", "AWARDS\nDean's list\nEDUCATION\n")
    headings = [s.heading for s in split_resume(text, words, max_tokens=200)]
    assert "AWARDS" not in headings  # 3 words: merged into EDUCATION
    education = next(s for s in split_resume(text, words, max_tokens=200) if s.heading == "EDUCATION")
    assert "Dean's list" in education.text


def test_unsectioned_text_falls_back_to_windows():
    text = "\n".join(f"line {i} of a plain resume with no headings at all" for i in range(30))
    sections = split_resume(text, words, max_tokens=50)
    assert len(sections) > 1
    assert all(s.heading.startswith("PART ") for s in sections)
    assert all(words(s.text) <= 50 for s in sections)


def test_single_overlong_line_is_split_on_words():
    text = "Header line\n" + " ".join(["word"] * 130)
    sections = split_resume(text, words, max_tokens=50)
    assert all(words(s.text) <= 50 for s in sections)
    assert sum(words(s.text) for s in sections) == 132


def test_empty_text():
    assert split_resume("  \n\n ", words) == []


def test_title_case_headings_outside_the_list_are_found():
    # Real: a resume using "Work History" / "Certifications and Tools" fell
    # back to one fixed window because neither heading was recognised.
    text = (
        "Priya Raman\n"
        "Work History\n" + bullets("Designed welding fixtures in SolidWorks", 3) + "\n"
        + "Certifications and Tools\n" + bullets("Lean Six Sigma Green Belt certified", 3)
    )
    headings = [s.heading for s in split_resume(text, words, max_tokens=200)]
    assert "WORK HISTORY" in headings and "CERTIFICATIONS AND TOOLS" in headings


def test_job_titles_and_sentences_are_not_headings():
    from resumematch.preprocess.resume_sections import _is_heading
    assert not _is_heading("Manufacturing Engineer II")   # Title Case, no keyword
    assert not _is_heading("Research Assistant")          # 'research' deliberately not a keyword
    assert not _is_heading("Gained hands-on experience")  # not Title Case
    assert not _is_heading("Strong Skills.")              # ends like a sentence
    assert _is_heading("Professional History:")
