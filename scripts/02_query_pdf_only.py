from __future__ import annotations

import argparse

from src.config.settings import Settings
from src.llm.openai_client import OpenAIClient
from src.retrieval.qdrant_store import get_client, search


def _snippet(text: str, n: int = 180) -> str:
    t = (text or "").replace("\n", " ").strip()
    return (t[:n] + "...") if len(t) > n else t


def main() -> None:
    parser = argparse.ArgumentParser(description="Query ONLY PDF chunks.")
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
        source="pdf",
        candidate_k=200,
    )

    print(f"\nQUERY (PDF ONLY): {args.query}\n")
    print("Top hits:\n")

    for i, (score, payload) in enumerate(hits, start=1):
        program_id = payload.get("program_id") or payload.get("doc_id") or "NA"
        title = payload.get("title") or "NA"
        chunk_id = payload.get("chunk_id") or "NA"
        text = payload.get("text") or ""

        print(f"{i}. score={score:.4f} | source=pdf | program_id={program_id} | title={title}")
        print(f"   chunk_id={chunk_id}")
        print(f"   {_snippet(text)}\n")


if __name__ == "__main__":
    main()
