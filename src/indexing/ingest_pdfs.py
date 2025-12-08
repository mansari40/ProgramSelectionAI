from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from pypdf import PdfReader

from src.utils.text import normalize_pdf_text


# Matches "Section Title ............ 12" (many dots + trailing page number)
_TOC_DOT_LEADER_LINE_RE = re.compile(r"^\s*.*\.{8,}\s*\d+\s*$")
# Matches lines that are only a page number (common in PDF extractions)
_PAGE_NUMBER_ONLY_RE = re.compile(r"^\s*\d+\s*$")


@dataclass(frozen=True)
class PDFDoc:
    doc_id: str
    title: str
    text: str
    meta: Dict[str, Any]


def _strip_toc_noise_preserve_lines(raw: str) -> str:
    """
    Remove common PDF TOC artifacts while line breaks still exist.
    This is intentionally conservative to avoid deleting real content.
    """
    if not raw:
        return ""

    out_lines: List[str] = []
    for line in raw.splitlines():
        s = line.strip()

        # Drop "Title ....... 12" style TOC lines
        if _TOC_DOT_LEADER_LINE_RE.match(s):
            continue

        # Drop standalone page numbers
        if _PAGE_NUMBER_ONLY_RE.match(s):
            continue

        out_lines.append(line)

    return "\n".join(out_lines).strip()


def _pdf_to_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))

    pages_text: List[str] = []
    for page in reader.pages:
        try:
            pages_text.append(page.extract_text() or "")
        except Exception:
            pages_text.append("")

    raw = "\n\n".join(pages_text)

    # 1) remove TOC dot-leader noise while lines still exist
    raw = _strip_toc_noise_preserve_lines(raw)

    # 2) then run your existing normalizer once
    return normalize_pdf_text(raw)


def _doc_id_from_filename(pdf_path: Path) -> str:
    # You are naming PDFs as Program_ID.pdf, so stem is already the ID.
    return pdf_path.stem


def ingest_pdf_dir(pdf_dir: Path) -> List[PDFDoc]:
    pdf_dir = Path(pdf_dir)
    if not pdf_dir.exists():
        return []

    pdfs = sorted(pdf_dir.glob("*.pdf"))

    docs: List[PDFDoc] = []
    for p in pdfs:
        doc_id = _doc_id_from_filename(p)
        title = p.name
        text = _pdf_to_text(p)

        meta: Dict[str, Any] = {
            "program_id": doc_id,
            "title": title,
            "path": str(p),
        }

        docs.append(PDFDoc(doc_id=doc_id, title=title, text=text, meta=meta))

    return docs
