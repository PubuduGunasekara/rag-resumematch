"""Turn a resume file (PDF or DOCX) into plain text.

Design note: keep this dumb on purpose. No layout reconstruction, no
column detection, just readable text. Chunking downstream is
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


class EncryptedPdfError(ValueError):
    """The PDF needs a password to read."""


def _parse_pdf(path: Path) -> str:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    if reader.is_encrypted:
        # Many "protected" PDFs only restrict editing and open with an
        # empty password; only a real open-password is a failure.
        try:
            opened = reader.decrypt("")
        except Exception:  # noqa: BLE001 - pypdf raises several types for unsupported encryption
            opened = 0
        if not opened:
            raise EncryptedPdfError(f"PDF is password-protected: {path.name}")
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _parse_docx(path: Path) -> str:
    """Paragraphs AND tables, in document order. Reading only
    doc.paragraphs dropped every table: a resume with its skills in a
    table lost them entirely (see docs/engineering-log.md, docx-tables).
    Each table row becomes one line, "Category: items" for two cells."""
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    doc = Document(str(path))
    lines: list[str] = []
    for child in doc.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            lines.append(Paragraph(child, doc).text)
        elif tag == "tbl":
            lines.extend(_table_lines(Table(child, doc)))
    return "\n".join(lines)


def _table_lines(table) -> list[str]:
    lines = []
    for row in table.rows:
        cells, seen = [], set()
        for cell in row.cells:
            if id(cell._tc) in seen:  # a merged cell is reported once per grid column
                continue
            seen.add(id(cell._tc))
            text = " ".join(cell.text.split())
            if text:
                cells.append(text)
        if cells:
            lines.append(f"{cells[0]}: {cells[1]}" if len(cells) == 2 else " | ".join(cells))
    return lines
