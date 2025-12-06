from qdrant_client.models import FieldCondition, Filter, MatchAny

from src.config.settings import Settings
from src.llm.openai_client import OpenAIClient
from src.retrieval.qdrant_store import get_client, search
from src.retrieval.program_index import load_program_index
from src.retrieval.program_resolver import resolve_program_ids


def build_filter_for_program_ids(program_ids: list[str]) -> Filter:
    return Filter(
        must=[
            FieldCondition(
                key="program_id",
                match=MatchAny(any=program_ids),
            )
        ]
    )


if __name__ == "__main__":
    cfg = Settings.load()
    client = get_client(cfg)
    llm = OpenAIClient(cfg)

    query = "What is the tuition fee for computer science?"
    q_emb = llm.embed_texts([query])[0]

    # 1) Resolve program name -> program_id (from Excel index)
    program_index = load_program_index(cfg)
    candidates = resolve_program_ids(query, program_index, max_results=5, min_score=0.50)
    program_ids = [c.program_id for c in candidates]

    # 2) Qdrant search (filtered if we found a confident program match)
    q_filter = build_filter_for_program_ids(program_ids) if program_ids else None
    hits = search(client, cfg.qdrant_collection, q_emb, top_k=12, query_filter=q_filter)

    print(f"QUERY: {query}\n")

    if program_ids:
        print("Resolved programs:")
        for c in candidates:
            print(f"- {c.program_id} | {c.title} | score={c.score:.2f}")
        print("\nTop hits (filtered to resolved programs):\n")
    else:
        print("Resolved programs: (none confident) — running unfiltered retrieval.\n")
        print("Top hits:\n")

    for i, (score, payload) in enumerate(hits, 1):
        program_id = payload.get("program_id") or payload.get("doc_id")
        title = payload.get("title", "")
        text = (payload.get("text") or "").replace("\n", " ").strip()
        print(f"{i}. score={score:.4f} | program_id={program_id} | title={title}")
        print(text[:220])
        print()
