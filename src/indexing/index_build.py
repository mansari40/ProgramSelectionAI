# src/indexing/index_build.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from src.config.settings import Settings
from src.indexing.chunking import ChunkingConfig, chunk_text
from src.indexing.ingest_excel import ingest_excel_as_program_chunks
from src.indexing.ingest_pdfs import ingest_pdf_dir
from src.indexing.ingest_web import ingest_web
from src.llm.openai_client import OpenAIClient
from src.retrieval.qdrant_store import (
    ensure_collection,
    get_client,
    points_exist,
    upsert_chunks,
)
from src.utils.ids import make_point_id
from src.utils.logger import get_logger, write_artifact

# Embedding model in Settings defaults to: text-embedding-3-large
# Its embedding dimension is 3072 (you verified via API call).
VECTOR_SIZE = 3072


@dataclass(frozen=True)
class IndexStats:
    docs: int
    chunks_total: int
    chunks_missing: int
    embeddings_computed: int
    points_upserted: int


def reset_collection(client, collection: str, vector_size: int = VECTOR_SIZE) -> None:
    """
    Destructive reset: deletes the collection if it exists, then recreates it.
    Use only when you intentionally want a clean rebuild (e.g., after corruption).
    """
    existing = [c.name for c in client.get_collections().collections]
    if collection in existing:
        client.delete_collection(collection)
    ensure_collection(client, collection, vector_size=vector_size)


# ----------------------------
# Shared helper (any chunks)
# ----------------------------
def _index_chunks(
    chunks: List,
    docs_count: int,
    artifact_name: str,
    *,
    reset: bool = False,
    force_upsert: bool = False,  # NEW: overwrite existing points too
) -> IndexStats:
    cfg = Settings.load()
    logger = get_logger()

    client = get_client(cfg)
    llm = OpenAIClient(cfg)

    # Ensure collection exists with correct vector size (or rebuild if reset=True)
    if reset:
        logger.info(f"RESET requested. Rebuilding collection: {cfg.qdrant_collection}")
        reset_collection(client, cfg.qdrant_collection, vector_size=VECTOR_SIZE)
    else:
        ensure_collection(client, cfg.qdrant_collection, vector_size=VECTOR_SIZE)

    point_ids = [make_point_id(c.chunk_id) for c in chunks]
    exists_flags = points_exist(client, cfg.qdrant_collection, point_ids)

    missing_chunks = [c for c, ex in zip(chunks, exists_flags) if not ex]
    to_upsert = chunks if force_upsert else missing_chunks

    logger.info(
        f"Docs: {docs_count} | chunks total: {len(chunks)} | "
        f"missing: {len(missing_chunks)} | to_upsert: {len(to_upsert)}"
    )

    embeddings = []
    if to_upsert:
        embeddings = llm.embed_texts([c.text for c in to_upsert])

    upserted = upsert_chunks(client, cfg.qdrant_collection, to_upsert, embeddings)

    stats = IndexStats(
        docs=docs_count,
        chunks_total=len(chunks),
        chunks_missing=len(missing_chunks),
        embeddings_computed=len(embeddings),
        points_upserted=upserted,
    )

    write_artifact(
        cfg.artifacts_dir,
        artifact_name,
        {
            "docs": stats.docs,
            "chunks_total": stats.chunks_total,
            "chunks_missing": stats.chunks_missing,
            "embeddings_computed": stats.embeddings_computed,
            "points_upserted": stats.points_upserted,
        },
    )

    logger.info(f"Upserted: {upserted} | Artifact written to {cfg.artifacts_dir}")
    return stats


# ----------------------------
# PDF indexing (Bachelor/Master)
# ----------------------------
def build_pdf_chunks(pdf_dir, cfg: Settings) -> Tuple[List, List]:
    docs = ingest_pdf_dir(pdf_dir)
    chunk_cfg = ChunkingConfig(
        chunk_chars=cfg.chunk_chars,
        overlap=cfg.chunk_overlap,
        max_chunks_per_doc=cfg.max_chunks_per_doc,
    )

    all_chunks = []
    for d in docs:
        meta = {
            "title": getattr(d, "title", None),
            "program_id": (getattr(d, "meta", None) or {}).get("program_id") or d.doc_id,
        }
        all_chunks.extend(
            chunk_text(
                source="pdf",
                doc_id=d.doc_id,
                text=d.text,
                cfg=chunk_cfg,
                meta=meta,
            )
        )
    return docs, all_chunks


def index_pdfs_bachelor(*, reset: bool = False) -> IndexStats:
    cfg = Settings.load()
    docs, chunks = build_pdf_chunks(cfg.pdf_bachelor_dir, cfg)
    return _index_chunks(
        chunks,
        docs_count=len(docs),
        artifact_name="index_pdfs_bachelor.json",
        reset=reset,
        force_upsert=False,
    )


def index_pdfs_master(*, reset: bool = False) -> IndexStats:
    cfg = Settings.load()
    docs, chunks = build_pdf_chunks(cfg.pdf_masters_dir, cfg)
    return _index_chunks(
        chunks,
        docs_count=len(docs),
        artifact_name="index_pdfs_master.json",
        reset=reset,
        force_upsert=False,
    )


# ----------------------------
# Excel indexing (Bachelor/Master)
# ----------------------------
def index_excel_bachelors(*, reset: bool = False) -> IndexStats:
    cfg = Settings.load()
    chunks = ingest_excel_as_program_chunks(cfg.bachelors_excel, excel_scope="bachelor")
    docs_count = len({c.doc_id for c in chunks})
    return _index_chunks(
        chunks,
        docs_count=docs_count,
        artifact_name="index_excel_bachelors.json",
        reset=reset,
        force_upsert=True,  # IMPORTANT: refresh Excel content even if points already exist
    )


def index_excel_masters(*, reset: bool = False) -> IndexStats:
    cfg = Settings.load()
    chunks = ingest_excel_as_program_chunks(cfg.masters_excel, excel_scope="master")
    docs_count = len({c.doc_id for c in chunks})
    return _index_chunks(
        chunks,
        docs_count=docs_count,
        artifact_name="index_excel_masters.json",
        reset=reset,
        force_upsert=True,  # IMPORTANT: refresh Excel content even if points already exist
    )


def index_excels_all(*, reset: bool = False) -> IndexStats:
    """
    Convenience: index both excels in one call.
    """
    cfg = Settings.load()
    b = ingest_excel_as_program_chunks(cfg.bachelors_excel, excel_scope="bachelor")
    m = ingest_excel_as_program_chunks(cfg.masters_excel, excel_scope="master")
    chunks = b + m
    docs_count = len({c.doc_id for c in chunks})
    return _index_chunks(
        chunks,
        docs_count=docs_count,
        artifact_name="index_excel_all.json",
        reset=reset,
        force_upsert=True,  # IMPORTANT: refresh Excel content even if points already exist
    )


# ----------------------------
# Website indexing (crawl -> extract -> chunk -> index)
# ----------------------------
def build_web_chunks(cfg: Settings) -> Tuple[List, List]:
    """
    ingest_web returns WebDoc objects with .doc_id .title .text .meta
    """
    docs = ingest_web(cfg)
    chunk_cfg = ChunkingConfig(
        chunk_chars=cfg.chunk_chars,
        overlap=cfg.chunk_overlap,
        max_chunks_per_doc=cfg.max_chunks_per_doc,
    )

    all_chunks = []
    for d in docs:
        meta = {"title": getattr(d, "title", None), **(getattr(d, "meta", None) or {})}
        all_chunks.extend(
            chunk_text(
                source="web",
                doc_id=d.doc_id,
                text=d.text,
                cfg=chunk_cfg,
                meta=meta,
            )
        )
    return docs, all_chunks


def index_web(*, reset: bool = False) -> IndexStats:
    cfg = Settings.load()
    logger = get_logger()

    if not getattr(cfg.web, "enabled", False):
        logger.info("Web ingestion disabled in config.yaml (web.enabled=false). Skipping.")
        return IndexStats(
            docs=0,
            chunks_total=0,
            chunks_missing=0,
            embeddings_computed=0,
            points_upserted=0,
        )

    docs, chunks = build_web_chunks(cfg)
    return _index_chunks(
        chunks,
        docs_count=len(docs),
        artifact_name="index_web.json",
        reset=reset,
        force_upsert=False,
    )


if __name__ == "__main__":
    # Rebuild behavior:
    # - PDFs/Web: upsert only missing (fast)
    # - Excel: force upsert (so schema/text improvements propagate)
    index_pdfs_master(reset=False)
    index_pdfs_bachelor(reset=False)
    index_excel_masters(reset=False)
    index_excel_bachelors(reset=False)
