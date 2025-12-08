from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Distance, Filter, PointStruct, VectorParams

from src.config.settings import Settings
from src.schemas import Chunk
from src.utils.ids import make_point_id


def get_client(cfg: Optional[Settings] = None) -> QdrantClient:
    cfg = cfg or Settings.load()
    return QdrantClient(url=cfg.qdrant_url, api_key=cfg.qdrant_api_key)


def ensure_collection(client: QdrantClient, collection: str, vector_size: int) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if collection in existing:
        return
    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )


def points_exist(
    client: QdrantClient,
    collection: str,
    point_ids: List[str],
    batch_size: int = 256,
) -> List[bool]:
    if not point_ids:
        return []
    flags: List[bool] = []
    for start in range(0, len(point_ids), batch_size):
        batch = point_ids[start : start + batch_size]
        res = client.retrieve(
            collection_name=collection,
            ids=batch,
            with_payload=False,
            with_vectors=False,
        )
        present = {p.id for p in res}
        flags.extend([pid in present for pid in batch])
    return flags


def upsert_chunks(
    client: QdrantClient,
    collection: str,
    chunks: Iterable[Chunk],
    embeddings: Iterable[List[float]],
    batch_size: int = 64,
) -> int:
    chunks_list = list(chunks)
    emb_list = list(embeddings)
    if len(chunks_list) != len(emb_list):
        raise ValueError("chunks and embeddings must have the same length")

    total = 0
    for start in range(0, len(chunks_list), batch_size):
        batch_chunks = chunks_list[start : start + batch_size]
        batch_embs = emb_list[start : start + batch_size]
        points: List[PointStruct] = []
        for ch, emb in zip(batch_chunks, batch_embs):
            pid = make_point_id(ch.chunk_id)
            payload = {
                "source": ch.source,
                "doc_id": ch.doc_id,
                "chunk_id": ch.chunk_id,
                "text": ch.text,
                **(ch.meta or {}),
            }
            points.append(PointStruct(id=pid, vector=emb, payload=payload))
        if points:
            client.upsert(collection_name=collection, points=points)
            total += len(points)
            print(f"[qdrant] upserted {total}/{len(chunks_list)}", flush=True)
    return total


# -----------------------------
# Search (robust / no server-side filters)
# -----------------------------


def _extract_source_from_filter(query_filter: Optional[Filter]) -> Optional[str]:
    """Best-effort extraction if caller passed a Qdrant Filter on source."""
    if query_filter is None:
        return None
    must = getattr(query_filter, "must", None)
    if not must:
        return None
    for cond in must:
        key = getattr(cond, "key", None)
        if key != "source":
            continue
        match = getattr(cond, "match", None)
        if match is None:
            continue
        value = getattr(match, "value", None)
        if isinstance(value, str) and value:
            return value.lower().strip()
    return None


def _local_filter_by_source(rows: List[Tuple[float, Dict[str, Any]]], source: str) -> List[Tuple[float, Dict[str, Any]]]:
    s = source.lower().strip()
    return [(score, payload) for score, payload in rows if (payload.get("source") or "").lower() == s]


def _unfiltered_vector_search(
    client: QdrantClient,
    collection: str,
    query_embedding: List[float],
    limit: int,
) -> List[Tuple[float, Dict[str, Any]]]:
    """Do NOT pass any filter into Qdrant (stability)."""
    if hasattr(client, "query_points"):
        res = client.query_points(
            collection_name=collection,
            query=query_embedding,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        points = getattr(res, "points", res)
        return [(p.score, p.payload or {}) for p in points]

    if hasattr(client, "search_points"):
        hits = client.search_points(
            collection_name=collection,
            query_vector=query_embedding,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return [(h.score, h.payload or {}) for h in hits]

    if hasattr(client, "search"):
        hits = client.search(
            collection_name=collection,
            query_vector=query_embedding,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return [(h.score, h.payload or {}) for h in hits]

    raise AttributeError("No supported search method found on QdrantClient.")


def search(
    client: QdrantClient,
    collection: str,
    query_embedding: List[float],
    top_k: int,
    query_filter: Optional[Filter] = None,
    source: Optional[str] = None,
    candidate_k: Optional[int] = None,
) -> List[Tuple[float, dict]]:
    """
    Robust search:
      1) Retrieve UNFILTERED candidates from Qdrant
      2) Apply local filtering (e.g., source=web/pdf/excel)
      3) Return top_k

    Key update: allow larger candidate pools (useful when you have many PDFs).
    """
    src = (source or "").strip().lower() or _extract_source_from_filter(query_filter)

    # Bigger candidate pools improve recall (especially for program-specific questions).
    # Keep it bounded to avoid extreme latency.
    cap = 2000
    base = top_k * 20  # higher than 10x because your corpus is growing
    cand = int(candidate_k or max(128, min(cap, base)))

    last_exc: Optional[Exception] = None
    # Retry strategy: progressively smaller limits if Qdrant returns instability.
    for lim in (cand, max(256, cand // 2), max(128, cand // 4), max(64, top_k * 10), top_k):
        try:
            rows = _unfiltered_vector_search(client, collection, query_embedding, limit=lim)
            if src:
                rows = _local_filter_by_source(rows, src)
            return rows[:top_k]
        except UnexpectedResponse as e:
            last_exc = e
            continue
        except Exception as e:
            last_exc = e
            continue

    if last_exc is not None:
        raise last_exc
    return []
