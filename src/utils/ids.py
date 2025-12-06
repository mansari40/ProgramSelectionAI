from __future__ import annotations

import uuid


def make_chunk_id(source: str, doc_id: str, chunk_index: int) -> str:
    return f"{source}::{doc_id}::{chunk_index}"


def make_point_id(chunk_id: str) -> str:
    # Deterministic UUID derived from chunk_id (stable across runs; Qdrant-compatible)
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))
