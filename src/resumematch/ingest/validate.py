"""Is this upload a resume we can analyze? Checked before any search or
LLM call, so a wrong file gets a specific, plain answer instead of a
traceback or a confident report about nothing.

Layers, in order (the first failure wins):
  1. File:   really a PDF / Word file underneath, not just named like one.
  2. Opens:  not damaged, not password-protected.
  3. Text:   has a text layer (a scanned image has none; no OCR here).
  4. Resume: structural signals a resume has and other documents lack.

1-3 are hard failures: there is nothing to analyze. 4 is soft: the
signals are heuristics, an unusual resume could miss them, and blocking
a real person's resume is worse than letting them continue after a clear
warning ("Check it anyway").

Why structure and not similarity to postings: measured on 9 documents,
a cover letter (0.710) and a job posting (0.806) were MORE similar to
postings than any real resume (0.556 to 0.687), because they are made of
job language. Headings, contact details and dated roles separated all 9.
See docs/engineering-log.md, "resume-check".
"""

import io
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from resumematch.preprocess.resume_sections import split_resume

MIN_WORDS = 40  # below this there is no resume to read (a scanned PDF gives 0)


@dataclass
class ResumeSignals:
    words: int
    headings: int        # sections found under a recognized heading
    contact: bool        # email, phone or LinkedIn
    dated_roles: int     # "2021 to Present", "2018–2021", "Jan 2020 - Mar 2022"
    job_ad_phrases: int  # "responsibilities", "you will", "we are looking for"

    @property
    def score(self) -> int:
        return (self.headings >= 2) + self.contact + (self.dated_roles >= 1)

    def missing(self) -> list[str]:
        out = []
        if not self.contact:
            out.append("contact details")
        if self.headings < 2:
            out.append("section headings such as Experience or Education")
        if self.dated_roles < 1:
            out.append("dates for roles or studies")
        return out


@dataclass
class ResumeRejected(Exception):
    code: str            # not_pdf | not_docx | unreadable | locked | no_text | job_posting | not_resume
    message: str         # shown to the user as is
    soft: bool = False   # True: the user may choose to continue
    signals: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return self.message


_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[a-z]{2,}", re.IGNORECASE)
_PHONE = re.compile(r"(\+\d{1,3}[\s.-]?)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}")
_LINKEDIN = re.compile(r"linkedin\.com/", re.IGNORECASE)
_MONTH = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"
_DATE = rf"(({_MONTH}|\d{{1,2}}/)\s*)?(19|20)\d{{2}}"
_DATED_ROLE = re.compile(rf"\b{_DATE}\s*(–|—|-|to|until)\s*({_DATE}|present|current|now|today)\b", re.IGNORECASE)
_JOB_AD = re.compile(
    r"\b(we are (looking|seeking|hiring)|you will|the ideal candidate|responsibilities include|"
    r"what you.ll (do|bring)|who you are|apply now|job description|about the role|"
    r"key responsibilities|we offer|equal opportunity employer)\b", re.IGNORECASE)


def check_file(path: str | Path) -> None:
    """Layer 1: the bytes match the extension. A text file renamed to
    .pdf used to reach the PDF parser and crash with PdfStreamError."""
    path = Path(path)
    data = path.read_bytes()
    suffix = path.suffix.lower()
    if suffix == ".pdf" and b"%PDF-" not in data[:1024]:
        raise ResumeRejected("not_pdf", "This file isn't a real PDF, even though its name ends in .pdf. "
                                        "Export your resume as a PDF from Word or Google Docs and try again.")
    if suffix == ".docx":
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                ok = "word/document.xml" in z.namelist()
        except zipfile.BadZipFile:
            ok = False
        if not ok:
            raise ResumeRejected("not_docx", "This file isn't a real Word document, even though its name ends "
                                             "in .docx. Save it from Word as .docx, or export it as a PDF.")


def extract_text(path: str | Path) -> str:
    """Layer 2: open it. Parser errors become plain messages; this is the
    one place that catches broadly, because it is the boundary with
    arbitrary uploaded bytes."""
    from resumematch.ingest.parse_resume import EncryptedPdfError, parse_resume
    try:
        return parse_resume(path)
    except EncryptedPdfError:
        raise ResumeRejected("locked", "This PDF is password-protected. Export an unlocked copy and upload that.")
    except Exception:  # noqa: BLE001 - pypdf / python-docx / zipfile raise many unrelated types
        raise ResumeRejected("unreadable", "This file couldn't be opened. It may be damaged. "
                                           "Export a fresh copy and try again.")


def resume_signals(text: str) -> ResumeSignals:
    sections = split_resume(text, lambda t: len(t.split()), max_tokens=100_000)
    headings = sum(not s.heading.startswith(("HEADER", "PART")) for s in sections)
    return ResumeSignals(
        words=len(text.split()),
        headings=headings,
        contact=bool(_EMAIL.search(text) or _PHONE.search(text) or _LINKEDIN.search(text)),
        dated_roles=len(_DATED_ROLE.findall(text)),
        job_ad_phrases=len(_JOB_AD.findall(text)),
    )


def check_text(text: str, allow_unusual: bool = False) -> ResumeSignals:
    """Layers 3 and 4. Returns the signals when the text passes."""
    signals = resume_signals(text)
    if signals.words < MIN_WORDS:
        raise ResumeRejected("no_text", "We couldn't find enough text in this file. If it's a scan or a photo, "
                                        "export your resume as a text PDF from Word or Google Docs.",
                             signals=signals.__dict__)
    if allow_unusual:
        return signals
    # A job-ad phrase in a document that also lacks a resume's structure.
    # (A real LinkedIn posting had one such phrase and score 0.)
    if signals.job_ad_phrases >= 1 and signals.score < 2:
        raise ResumeRejected("job_posting", "This looks like a job posting, not a resume. Upload your own resume "
                                            "to compare it with postings.", soft=True, signals=signals.__dict__)
    if signals.score < 2:
        missing = signals.missing()
        listed = (missing[0] if len(missing) == 1 else f"{missing[0]} or {missing[1]}" if len(missing) == 2
                  else ", ".join(missing[:-1]) + ", or " + missing[-1])
        raise ResumeRejected("not_resume", f"This doesn't look like a resume. We couldn't find {listed}. "
                                           "If it is your resume, you can check it anyway.",
                             soft=True, signals=signals.__dict__)
    return signals


def validate_resume(path: str | Path, allow_unusual: bool = False) -> tuple[str, ResumeSignals]:
    """All four layers. Returns (text, signals) or raises ResumeRejected."""
    check_file(path)
    text = extract_text(path)
    return text, check_text(text, allow_unusual=allow_unusual)
