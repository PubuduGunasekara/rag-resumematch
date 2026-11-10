"""Parsing and section detection on the synthetic resumes in eval/resumes/.

The 17 numbered resumes are fictional (example.com emails, 555 phone
numbers); see eval/resumes/MANIFEST.md. Each test here started as a bug
seen on one of them.
"""

from pathlib import Path

import pytest

from resumematch.ingest.parse_resume import parse_resume
from resumematch.preprocess.resume_sections import split_resume

RESUMES = Path("eval/resumes")


def words(text: str) -> int:
    return len(text.split())


def headings(path: Path) -> list[str]:
    return [s.heading for s in split_resume(parse_resume(path), words, max_tokens=480)]


def test_docx_table_cells_are_read():
    # Bug: parse_resume read only paragraphs, so the skills table in this
    # resume vanished (194 words vs 240 in the PDF; 'VMware' missing).
    docx = parse_resume(RESUMES / "12_sysadmin_helpdesk_typos.docx")
    pdf = parse_resume(RESUMES / "12_sysadmin_helpdesk_typos.pdf")
    assert "VMware" in docx
    assert abs(words(docx) - words(pdf)) <= 10
    assert headings(RESUMES / "12_sysadmin_helpdesk_typos.docx") == headings(RESUMES / "12_sysadmin_helpdesk_typos.pdf")


def test_docx_table_keeps_document_order():
    # The table sits between the summary and the experience section; its
    # text must come out there, not appended at the end.
    text = parse_resume(RESUMES / "12_sysadmin_helpdesk_typos.docx")
    lines = text.splitlines()
    summary = next(i for i, line in enumerate(lines) if line.strip().upper() == "SUMMARY")
    vmware = next(i for i, line in enumerate(lines) if "VMware" in line)
    experience = next(i for i, line in enumerate(lines) if line.strip().upper() == "EXPERIENCE")
    assert summary < vmware < experience


def test_skill_categories_in_a_pdf_sidebar_are_not_headings():
    # Bug: in a two-column PDF the sidebar comes out first, and category
    # labels inside Skills ('Languages', 'LLM') started new sections, which
    # also swallowed the real 'Skills' heading.
    found = headings(RESUMES / "04_ml_engineer_llm.pdf")
    assert "SKILLS" in found
    assert "LANGUAGES" not in found and "LLM" not in found
    assert {"SUMMARY", "EXPERIENCE", "PROJECTS"} <= set(found)


@pytest.mark.parametrize("path", sorted(RESUMES.glob("[0-1][0-9]_*.*")), ids=lambda p: p.name)
def test_every_synthetic_resume_parses_into_several_sections(path):
    text = parse_resume(path)
    assert words(text) > 100
    assert len(split_resume(text, words, max_tokens=480)) >= 3
