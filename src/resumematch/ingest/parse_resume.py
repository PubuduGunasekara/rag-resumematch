"""Turn a resume file (PDF or DOCX) into plain text.

Design note: keep this dumb on purpose. No layout reconstruction, no
column detection — just get readable text out. Chunking downstream is
what has to be smart, not this.
"""

from pathlib import Path


def parse_resume(file_path: str | Path) -> str:
    """
    Input:
        file_path: path to a .pdf or .docx resume file.
    Output:
        The resume's text content as a single string, with page/section
        boundaries preserved as newlines (not stripped into one blob).
    Raises:
        ValueError: if the file extension isn't .pdf or .docx.
        FileNotFoundError: if file_path doesn't exist.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Resume file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _parse_pdf(path)
    elif suffix == ".docx":
        return _parse_docx(path)
    else:
        raise ValueError(f"Unsupported resume format: {suffix} (expected .pdf or .docx)")


def _parse_pdf(path: Path) -> str:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _parse_docx(path: Path) -> str:
    from docx import Document
    doc = Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs)
