from __future__ import annotations

import argparse
from typing import Set

from qdrant_client.models import Filter, FieldCondition, MatchValue

from src.config.settings import Settings
from src.llm.openai_client import OpenAIClient
from src.retrieval.qdrant_store import get_client, search


def _snippet(text: str, n: int = 180) -> str:
    t = (text or "").replace("\n", " ").strip()
    return (t[:n] + "...") if len(t) > n else t


def _mk_source_filter(source: str) -> Filter | None:
    source = (source or "all").lower().strip()
    if source in ("all", "*"):
        return None
    if source not in ("pdf", "web", "excel"):
        raise ValueError("source must be one of: all, pdf, web, excel")
    return Filter(
        must=[
            FieldCondition(
                key="source",
                match=MatchValue(value=source),
            )
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the full Constructor KB (pdf + excel + web).")
    parser.add_argument("--q", "--query", dest="query", required=True, help="Your question")
    parser.add_argument("--topk", type=int, default=12, help="Number of results")
    parser.add_argument("--source", type=str, default="all", help="all|pdf|web|excel (default: all)")
    args = parser.parse_args()

    cfg = Settings.load()
    client = get_client(cfg)
    llm = OpenAIClient(cfg)

    q_emb = llm.embed_texts([args.query])[0]
    f = _mk_source_filter(args.source)

    hits = search(client, cfg.qdrant_collection, q_emb, top_k=args.topk, query_filter=f)

    print(f"\nQUERY: {args.query}")
    print(f"SOURCE FILTER: {args.source}\n")
    print("Top hits:\n")

    seen: Set[str] = set()
    shown = 0

    for score, payload in hits:
        # de-dupe by chunk_id (preferred), otherwise by doc_id+text snippet prefix
        chunk_id = payload.get("chunk_id") or ""
        dedupe_key = chunk_id or (str(payload.get("doc_id")) + "::" + _snippet(payload.get("text", ""), 80))
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        source = payload.get("source", "unknown")
        program_id = payload.get("program_id") or payload.get("doc_id") or "NA"
        doc_id = payload.get("doc_id") or "NA"
        title = payload.get("title") or "NA"
        program_name = payload.get("program_name") or "NA"

        # web-specific
        url = payload.get("url") or payload.get("source_url") or "NA"

        text = payload.get("text") or ""

        shown += 1
        print(f"{shown}. score={score:.4f} | source={source} | program_id={program_id} | doc_id={doc_id}")
        if source == "web":
            print(f"   url={url}")
            print(f"   title={title}")
        elif source == "excel":
            print(f"   program_name={program_name}")
            print(f"   title={title}")
        else:  # pdf (or unknown)
            print(f"   title={title}")
        print(f"   chunk_id={chunk_id or 'NA'}")
        print(f"   {_snippet(text)}\n")

        if shown >= args.topk:
            break


if __name__ == "__main__":
    main()
