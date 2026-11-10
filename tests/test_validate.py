"""Upload validation: every kind of wrong file gets the right, plain
answer, and real resumes are never blocked."""

from pathlib import Path

import pytest
from docx import Document
from pypdf import PdfWriter

from resumematch.ingest.validate import ResumeRejected, check_text, resume_signals, validate_resume

RESUMES = sorted(Path("eval/resumes").glob("*.docx"))


def docx(path: Path, lines: list[str]) -> Path:
    d = Document()
    for line in lines:
        d.add_paragraph(line)
    d.save(path)
    return path


def rejected(path: Path, **kw) -> ResumeRejected:
    with pytest.raises(ResumeRejected) as info:
        validate_resume(path, **kw)
    return info.value


@pytest.mark.parametrize("path", RESUMES, ids=lambda p: p.stem)
def test_real_resumes_pass(path):
    text, signals = validate_resume(path)
    assert signals.score >= 2 and len(text.split()) > 100


def test_text_file_renamed_to_pdf(tmp_path):
    # Real: crashed the PDF parser with PdfStreamError (a 500 / traceback).
    p = tmp_path / "cv.pdf"
    p.write_text("just some text, not a pdf")
    e = rejected(p)
    assert e.code == "not_pdf" and not e.soft


def test_broken_docx(tmp_path):
    p = tmp_path / "cv.docx"
    p.write_bytes(b"PK\x03\x04garbage")
    assert rejected(p).code == "not_docx"


def test_zip_that_is_not_a_word_document(tmp_path):
    import zipfile
    p = tmp_path / "cv.docx"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("hello.txt", "hi")
    assert rejected(p).code == "not_docx"


def test_password_protected_pdf(tmp_path):
    p = tmp_path / "cv.pdf"
    w = PdfWriter()
    w.add_blank_page(612, 792)
    w.encrypt("secret")
    w.write(p)
    e = rejected(p)
    assert e.code == "locked" and "password" in e.message


def test_pdf_with_no_text_layer_like_a_scan(tmp_path):
    p = tmp_path / "scan.pdf"
    w = PdfWriter()
    w.add_blank_page(612, 792)
    w.write(p)
    assert rejected(p).code == "no_text"


def test_damaged_pdf_with_a_valid_header(tmp_path):
    p = tmp_path / "cv.pdf"
    p.write_bytes(b"%PDF-1.7\n" + b"\x00garbage" * 50)
    assert rejected(p).code in ("unreadable", "no_text")


def test_recipe_is_not_a_resume_but_can_be_checked_anyway(tmp_path):
    p = docx(tmp_path / "recipe.docx", ["Grandma's banana bread", "Ingredients", "3 ripe bananas, mashed",
             "Preheat the oven to 350F and butter a loaf pan. Mix the bananas and butter, then stir in the sugar, "
             "egg and vanilla. Bake for 60 minutes until a tester comes out clean. Let it cool before slicing it."])
    e = rejected(p)
    assert e.code == "not_resume" and e.soft
    assert ("It has no contact details, section headings such as Experience or Education, "
            "or dates for roles or studies.") in e.message and "check it anyway" in e.message
    text, _ = validate_resume(p, allow_unusual=True)   # the user's override
    assert "banana" in text


COVER_LETTER = ("Dear Hiring Manager,\nI am writing to apply for the Staff Accountant position at your company. "
                "With four years of experience in month-end close and account reconciliation, I believe I would be "
                "a strong addition to your team. I am a CPA candidate and an advanced Excel user. Thank you for your "
                "time and consideration.\nSincerely,\nJames Okafor\njames.okafor@example.com | (312) 555-0199")
JOB_POSTING = ("Staff Accountant\nWe are looking for a detail-oriented Staff Accountant to join our finance team. "
               "You will own the month-end close, reconcile balance sheet accounts and support the annual audit. "
               "Requirements: a bachelor's degree in accounting, 2+ years of experience, and strong Excel skills. "
               "We offer competitive pay, medical, dental and vision coverage, and a hybrid schedule.")


def test_cover_letter_is_flagged_even_though_it_has_contact_details():
    # A cover letter scored HIGHER similarity to postings (0.710) than any
    # real resume; only structure tells them apart.
    with pytest.raises(ResumeRejected) as info:
        check_text(COVER_LETTER)
    assert info.value.code == "not_resume"


def test_job_posting_gets_its_own_message():
    with pytest.raises(ResumeRejected) as info:
        check_text(JOB_POSTING)
    assert info.value.code == "job_posting" and "job posting" in info.value.message


@pytest.mark.parametrize("line", ["Engineer (2021–Present)", "Analyst, Jan 2020 - Mar 2022", "2018 to 2021",
                                  "06/2019 - 08/2021", "Intern, Summer 2023 – current"])
def test_dated_roles_in_common_formats(line):
    assert resume_signals(line).dated_roles == 1


def test_single_years_are_not_dated_roles():
    assert resume_signals("Construction is expected to begin in 2025.").dated_roles == 0
