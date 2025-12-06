from __future__ import annotations

import argparse

from src.config.settings import Settings
from src.llm.openai_client import OpenAIClient
from src.retrieval.qdrant_store import get_client, search


def _snippet(text: str, n: int = 180) -> str:
    t = (text or "").replace("\n", " ").strip()
    return (t[:n] + "...") if len(t) > n else t


def main() -> None:
    parser = argparse.ArgumentParser(description="Query ONLY web-ingested pages.")
    parser.add_argument("--q", "--query", dest="query", required=True, help="Your question")
    parser.add_argument("--topk", type=int, default=12, help="Number of results")
    args = parser.parse_args()

    cfg = Settings.load()
    client = get_client(cfg)
    llm = OpenAIClient(cfg)

    q_emb = llm.embed_texts([args.query])[0]

    hits = search(
        client,
        cfg.qdrant_collection,
        q_emb,
        top_k=args.topk,
        source="web",          # local filtering (stable)
        candidate_k=200,       # more candidates helps after filtering
    )

    print(f"\nQUERY (WEB ONLY): {args.query}\n")
    print("Top hits:\n")

    for i, (score, payload) in enumerate(hits, start=1):
        doc_id = payload.get("doc_id") or "NA"
        title = payload.get("title") or payload.get("program_name") or "NA"
        chunk_id = payload.get("chunk_id") or "NA"
        text = payload.get("text") or ""

        print(f"{i}. score={score:.4f} | source=web | doc_id={doc_id}")
        print(f"   title/name={title}")
        print(f"   chunk_id={chunk_id}")
        print(f"   {_snippet(text)}\n")


if __name__ == "__main__":
    main()
