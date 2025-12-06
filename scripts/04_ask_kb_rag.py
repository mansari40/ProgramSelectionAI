from __future__ import annotations

import argparse
from typing import List, Tuple, Optional

from qdrant_client.models import Filter, FieldCondition, MatchValue

from src.config.settings import Settings
from src.llm.openai_client import OpenAIClient
from src.retrieval.qdrant_store import get_client, search


SYSTEM_PROMPT = """You are a retrieval-augmented assistant for Constructor University.
You MUST follow these rules:
1) Use ONLY the provided context excerpts as evidence.
2) Do NOT assume anything not present in the excerpts.
3) If the excerpts do not contain the answer, say: "I don't know based on the provided documents."
4) When the question compares two programs, you must show evidence for BOTH; if evidence for one side is missing, say so.
5) End with a "Citations" section listing bullet points in this format:
   - program_id | chunk_id | short quote (<=12 words)
"""


def _mk_source_filter(source: str) -> Optional[Filter]:
    s = (source or "all").lower().strip()
    if s in ("all", "*"):
        return None
    if s not in ("pdf", "web", "excel"):
        raise ValueError("Invalid --source. Use one of: all, pdf, web, excel.")
    return Filter(
        must=[
            FieldCondition(
                key="source",
                match=MatchValue(value=s),
            )
        ]
    )


def build_context(hits: List[Tuple[float, dict]], max_chars: int = 12000) -> str:
    """
    Build an evidence pack from retrieved chunks.
    Keep it bounded so we don't blow up the prompt.
    """
    parts: List[str] = []
    used = 0

    for score, payload in hits:
        source = payload.get("source", "unknown")
        program_id = payload.get("program_id") or payload.get("doc_id") or "NA"
        title = payload.get("title") or payload.get("program_name") or "NA"
        chunk_id = payload.get("chunk_id") or "NA"
        url = payload.get("url") or payload.get("source_url") or ""

        text = (payload.get("text") or "").strip()
        if not text:
            continue

        header = f"[source={source} | program_id={program_id} | chunk_id={chunk_id} | score={score:.4f} | title={title}"
        if url:
            header += f" | url={url}"
        header += "]\n"

        block = header + text + "\n"
        if used + len(block) > max_chars:
            break

        parts.append(block)
        used += len(block)

    return "\n---\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the Constructor KB using RAG (pdf + excel + web).")
    parser.add_argument("--q", "--query", dest="query", required=True, help="Your question")
    parser.add_argument("--topk", type=int, default=12, help="How many chunks to retrieve")
    parser.add_argument("--source", type=str, default="all", help="all|pdf|web|excel (default: all)")
    parser.add_argument("--max_context_chars", type=int, default=12000, help="Max characters packed into context")
    args = parser.parse_args()

    cfg = Settings.load()
    client = get_client(cfg)
    llm = OpenAIClient(cfg)

    # 1) Embed query
    q_emb = llm.embed_texts([args.query])[0]

    # 2) Retrieve
    f = _mk_source_filter(args.source)
    hits = search(
        client,
        cfg.qdrant_collection,
        q_emb,
        top_k=args.topk,
        query_filter=f,
    )

    # 3) Build context
    context = build_context(hits, max_chars=args.max_context_chars)

    # 4) Compose prompt
    user_prompt = f"""QUESTION:
{args.query}

CONTEXT EXCERPTS:
{context}
"""

    # 5) LLM answer
    answer = llm.chat(SYSTEM_PROMPT, user_prompt)

    print("\nQUESTION:", args.query)
    print("SOURCE FILTER:", args.source)
    print("\nANSWER:\n", answer)


if __name__ == "__main__":
    main()
