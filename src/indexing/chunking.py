from __future__ import annotations

from dataclasses import dataclass
from typing import List

from src.schemas import Chunk
from src.utils.ids import make_chunk_id


@dataclass(frozen=True)
class ChunkingConfig:
    chunk_chars: int = 2800
    overlap: int = 350
    max_chunks_per_doc: int = 400


def chunk_text(
    *,
    source: str,
    doc_id: str,
    text: str,
    cfg: ChunkingConfig,
    meta: dict | None = None,
) -> List[Chunk]:
    """
    Simple character-based chunking with overlap.
    Keeps chunk boundaries stable and reproducible.
    """
    meta = meta or {}
    text = (text or "").strip()
    if not text:
        return []

    chunks: List[Chunk] = []
    start = 0
    i = 0

    while start < len(text) and i < cfg.max_chunks_per_doc:
        end = min(start + cfg.chunk_chars, len(text))
        chunk_str = text[start:end].strip()
        if chunk_str:
            chunks.append(
                Chunk(
                    source=source,  # "pdf" | "excel" | "web"
                    doc_id=doc_id,
                    chunk_id=make_chunk_id(source, doc_id, i),
                    text=chunk_str,
                    meta=dict(meta),
                )
            )
            i += 1

        if end >= len(text):
            break

        start = max(0, end - cfg.overlap)

    return chunks
