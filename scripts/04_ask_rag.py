from src.config.settings import Settings
from src.llm.openai_client import OpenAIClient
from src.retrieval.qdrant_store import get_client, search


SYSTEM_PROMPT = """You are a retrieval-augmented assistant for Constructor University.
You MUST follow these rules:
1) Use ONLY the provided context excerpts as evidence.
2) Do NOT infer, guess, or add "typical" subjects not stated in the excerpts.
3) If the excerpts do not contain the answer, say: "I don't know based on the provided documents."
4) If the question compares two programs, you must show evidence for BOTH. If evidence for one side is missing, say so.
5) End with a "Citations" section listing bullet points in this format:
   - program_id | chunk_id | short quote (<=12 words)
"""


def build_context(hits, max_chars: int = 12000) -> str:
    parts = []
    used = 0
    for score, payload in hits:
        program_id = payload.get("program_id") or payload.get("doc_id")
        chunk_id = payload.get("chunk_id", "")
        title = payload.get("title", "")
        text = (payload.get("text") or "").strip()

        block = (
            f"[program_id={program_id} | chunk_id={chunk_id} | title={title} | score={score:.4f}]\n"
            f"{text}\n"
        )

        if used + len(block) > max_chars:
            break

        parts.append(block)
        used += len(block)

    return "\n---\n".join(parts)


if __name__ == "__main__":
    cfg = Settings.load()
    client = get_client(cfg)
    llm = OpenAIClient(cfg)

    query = "Give me some information about scholarship and tuition for Bachelor in Computer science at constructor university?"
    q_emb = llm.embed_texts([query])[0]

    hits = search(client, cfg.qdrant_collection, q_emb, top_k=12)
    context = build_context(hits)

    user_prompt = f"""QUESTION:
{query}

CONTEXT EXCERPTS:
{context}
"""

    # IMPORTANT: match your OpenAIClient.chat(system_prompt, user_prompt)
    answer = llm.chat(SYSTEM_PROMPT, user_prompt)

    print("\nANSWER:\n", answer)
