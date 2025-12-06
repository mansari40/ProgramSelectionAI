from __future__ import annotations

import re
from typing import Iterable


_BR_TAG_RE = re.compile(r"(?i)<\s*br\s*/?\s*>|(?:^|\s)br>", re.MULTILINE)
_HTML_TAG_RE = re.compile(r"(?s)<[^>]+>")  # very light tag stripper
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_SOFT_HYPHEN_RE = re.compile(r"\u00ad")  # soft hyphen char
_PAGE_BREAK_RE = re.compile(r"\f")


def normalize_pdf_text(text: str) -> str:
    """
    Normalize typical PDF extraction artifacts:
    - remove <br>, br> and common HTML-like tags
    - remove soft hyphens
    - fix hyphenation across line breaks: "exam-\nination" -> "examination"
    - normalize line breaks and spaces
    """
    if not text:
        return ""

    # Standardize newlines
    t = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove form-feed page breaks
    t = _PAGE_BREAK_RE.sub("\n", t)

    # Remove soft hyphens
    t = _SOFT_HYPHEN_RE.sub("", t)

    # Remove <br> / br> artifacts
    t = _BR_TAG_RE.sub("\n", t)

    # Strip other HTML-ish tags if they appear
    # (PDF extractors sometimes emit fragments like <b> or <i> etc.)
    t = _HTML_TAG_RE.sub("", t)

    # Fix hyphenation at line wraps: "data-\nbase" -> "database"
    t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)

    # Convert single newlines inside paragraphs into spaces.
    # Keep paragraph breaks when there are 2+ newlines.
    # Strategy:
    # 1) temporarily mark paragraph boundaries
    t = t.replace("\n\n", "\n¶\n")
    # 2) flatten remaining newlines to spaces
    t = t.replace("\n", " ")
    # 3) restore paragraph boundaries
    t = t.replace(" ¶ ", "\n\n").replace("¶", "\n\n")

    # Normalize whitespace
    t = _MULTI_SPACE_RE.sub(" ", t).strip()
    t = _MULTI_NEWLINE_RE.sub("\n\n", t)

    return t


def normalize_program_name(name: str) -> str:
    """
    Light normalization to help matching user queries to program names.
    (Keeps it conservative to avoid breaking proper titles.)
    """
    if not name:
        return ""
    t = name.strip()
    t = _MULTI_SPACE_RE.sub(" ", t)
    return t


def join_nonempty(parts: Iterable[str], sep: str = "\n") -> str:
    items = [p.strip() for p in parts if p and p.strip()]
    return sep.join(items)
