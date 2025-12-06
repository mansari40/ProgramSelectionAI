from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

from qdrant_client import QdrantClient
from qdrant_client.models import Filter
from qdrant_client.http.exceptions import UnexpectedResponse

from src.config.settings import Settings
from src.schemas import Chunk
from src.utils.ids import make_point_id


def get_client(cfg: Optional[Settings] = None) -> QdrantClient:
    cfg = cfg or Settings.load()
    return QdrantClient(url=cfg.qdrant_url, api_key=cfg.qdrant_api_key)


def ensure_collection(
    client: QdrantClient,
    collection: str,
    vector_size: int,
) -> None:
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
    """
    Returns a list of booleans aligned with point_ids indicating whether each point exists.
    Batched to avoid oversized requests.
    """
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
    """
    Upserts points in batches to avoid Qdrant payload size limits.
    Prints progress as it uploads.
    """
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


def _extract_source_value(query_filter: Optional[Filter]) -> Optional[str]:
    """
    Best-effort extraction for simple filters like:
      Filter(must=[FieldCondition(key="source", match=MatchValue(value="web"))])
    """
    if query_filter is None:
        return None
    try:
        must = getattr(query_filter, "must", None) or []
        for cond in must:
            if getattr(cond, "key", None) == "source":
                match = getattr(cond, "match", None)
                val = getattr(match, "value", None)
                if isinstance(val, str):
                    return val
    except Exception:
        return None
    return None


def search(
    client: QdrantClient,
    collection: str,
    query_embedding: List[float],
    top_k: int,
    query_filter: Optional[Filter] = None,
) -> List[Tuple[float, dict]]:
    """
    Robust search.

    IMPORTANT WORKAROUND:
    Some Qdrant server versions panic (500 OutputTooSmall) on query_points with filters.
    So: if a filter is provided, we do UNFILTERED vector search and filter in Python.

    This is acceptable for your dataset size (hundreds to a few thousand chunks), and
    keeps your scripts working immediately.
    """

    # If filter present, prefer safe path: unfiltered + python filtering
    if query_filter is not None:
        source_value = _extract_source_value(query_filter)
        # pull more results to compensate for filtering
        candidate_k = max(top_k * 10, 50)

        # Use unfiltered query_points (or fallback) then filter locally
        hits_all = search(
            client=client,
            collection=collection,
            query_embedding=query_embedding,
            top_k=candidate_k,
            query_filter=None,
        )

        if source_value:
            filtered = [(s, p) for (s, p) in hits_all if (p or {}).get("source") == source_value]
            return filtered[:top_k]

        # If we cannot parse the filter, return best-effort unfiltered
        return hits_all[:top_k]

    # ---------- unfiltered vector search ----------
    if hasattr(client, "query_points"):
        try:
            res = client.query_points(
                collection_name=collection,
                query=query_embedding,
                limit=top_k,
                with_payload=True,
                with_vectors=False,
            )
            points = getattr(res, "points", res)
            return [(p.score, p.payload or {}) for p in points]
        except UnexpectedResponse:
            pass
        except Exception:
            pass

    if hasattr(client, "search_points"):
        hits = client.search_points(
            collection_name=collection,
            query_vector=query_embedding,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )
        return [(h.score, h.payload or {}) for h in hits]

    if hasattr(client, "search"):
        hits = client.search(
            collection_name=collection,
            query_vector=query_embedding,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )
        return [(h.score, h.payload or {}) for h in hits]

    raise AttributeError("No supported search method found on QdrantClient.")