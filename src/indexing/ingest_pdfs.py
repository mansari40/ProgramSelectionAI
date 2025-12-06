from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from pypdf import PdfReader

from src.utils.text import normalize_pdf_text


@dataclass(frozen=True)
class PDFDoc:
    doc_id: str
    title: str
    text: str
    meta: Dict[str, Any]


def _pdf_to_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages_text: List[str] = []
    for page in reader.pages:
        try:
            pages_text.append(page.extract_text() or "")
        except Exception:
            pages_text.append("")
    raw = "\n\n".join(pages_text)
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
