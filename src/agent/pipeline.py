from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

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

LIST_INTENT_RE = re.compile(
    r"^(list|show me|show|which programs|programs that|find programs|find|give me all|all programs)\b",
    re.IGNORECASE,
)


def _snippet(text: str, n: int = 220) -> str:
    t = (text or "").replace("\n", " ").strip()
    return (t[:n] + "...") if len(t) > n else t


def _short_quote(text: str, max_words: int = 12) -> str:
    words = (text or "").replace("\n", " ").strip().split()
    return " ".join(words[:max_words])


def _mode_for_query(query: str) -> str:
    return "list" if LIST_INTENT_RE.search((query or "").strip()) else "answer"


def _normalize_source(source: str) -> Optional[str]:
    """
    Returns:
      None  -> means "all"
      "pdf"|"web"|"excel" -> caller wants that source only
    """
    s = (source or "all").strip().lower()
    if s in ("all", "*"):
        return None
    if s not in ("pdf", "excel", "web"):
        raise ValueError("source must be one of: all, pdf, excel, web")
    return s


def _build_context(hits: List[Tuple[float, dict]], max_chars: int = 12000) -> str:
    parts: List[str] = []
    used = 0

    for score, payload in hits:
        source = payload.get("source", "unknown")
        program_id = payload.get("program_id") or payload.get("doc_id") or "NA"
        title = payload.get("title", "") or payload.get("program_name", "") or ""
        chunk_id = payload.get("chunk_id", "NA")
        text = (payload.get("text") or "").strip()

        block = (
            f"[source={source} | program_id={program_id} | chunk_id={chunk_id} | title={title} | score={score:.4f}]\n"
            f"{text}\n"
        )

        if used + len(block) > max_chars:
            break

        parts.append(block)
        used += len(block)

    return "\n---\n".join(parts)


def _extract_focus_terms(query: str) -> List[str]:
    """
    If the list query contains explicit terms, require evidence text to contain them.
    """
    q = (query or "").lower()
    terms: List[str] = []

    for t in ["python", "sql", "machine learning", "deep learning", "nlp", "data engineering", "data science"]:
        if t in q:
            terms.append(t)

    seen = set()
    out: List[str] = []
    for t in terms:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


def _contains_all_terms(text: str, terms: List[str]) -> bool:
    if not terms:
        return True
    low = (text or "").lower()
    return all(t in low for t in terms)


def _group_programs(hits: List[Tuple[float, dict]]) -> List[Dict[str, Any]]:
    """
    Group by program_id/doc_id and keep the best-scoring hit per program.
    """
    best_by_program: Dict[str, Tuple[float, dict]] = {}

    for score, payload in hits:
        pid = payload.get("program_id") or payload.get("doc_id") or "NA"
        if pid == "NA":
            continue
        if (pid not in best_by_program) or (score > best_by_program[pid][0]):
            best_by_program[pid] = (score, payload)

    grouped: List[Dict[str, Any]] = []
    for pid, (score, payload) in best_by_program.items():
        grouped.append(
            {
                "program_id": pid,
                "score": float(score),
                "source": payload.get("source", "unknown"),
                "program_name": payload.get("program_name") or payload.get("title") or pid,
                "chunk_id": payload.get("chunk_id") or "NA",
                "text": payload.get("text") or "",
            }
        )

    grouped.sort(key=lambda x: x["score"], reverse=True)
    return grouped


def ask_kb(query: str, source: str = "all", top_k: int = 12) -> Dict[str, Any]:
    cfg = Settings.load()
    client = get_client(cfg)
    llm = OpenAIClient(cfg)

    mode = _mode_for_query(query)
    q_emb = llm.embed_texts([query])[0]

    src = _normalize_source(source)

    # Retrieval sizing:
    # - answer-mode: pull enough for context (candidate_k helps Qdrant stability + local filtering)
    # - list-mode: pull more, then group by program_id/doc_id
    if mode == "list":
        retrieve_k = min(max(top_k * 10, 120), 250)
        candidate_k = 350
    else:
        retrieve_k = top_k
        candidate_k = max(200, top_k * 10)

    # IMPORTANT: do NOT pass Qdrant Filter objects. Use local filtering via `source=...`.
    hits = search(
        client,
        cfg.qdrant_collection,
        q_emb,
        top_k=retrieve_k,
        source=src,
        candidate_k=candidate_k,
        query_filter=None,  # keep explicit; prevents accidental server-side filtering
    )

    # For debugging / UI
    hits_view: List[Dict[str, Any]] = []
    for score, payload in hits:
        hits_view.append(
            {
                "score": float(score),
                "source": payload.get("source", "unknown"),
                "program_id": payload.get("program_id") or payload.get("doc_id") or "NA",
                "program_name": payload.get("program_name") or payload.get("title") or "NA",
                "chunk_id": payload.get("chunk_id") or "NA",
                "text_snippet": _snippet(payload.get("text") or ""),
            }
        )

    # ----------------
    # LIST MODE
    # ----------------
    if mode == "list":
        programs = _group_programs(hits)

        focus_terms = _extract_focus_terms(query)
        if focus_terms:
            programs = [p for p in programs if _contains_all_terms(p.get("text", ""), focus_terms)]

        if not programs:
            return {
                "mode": mode,
                "answer": "I don't know based on the provided documents.",
                "citations": [],
                "hits": hits_view,
            }

        keep_n = max(10, min(40, top_k * 4))
        programs = programs[:keep_n]

        lines = ["Programs found (best evidence per program):", ""]
        for i, p in enumerate(programs, start=1):
            lines.append(f"{i}. {p['program_name']} ({p['program_id']}) [{p['source']}]")

        lines.append("")
        lines.append("Citations:")

        citations_struct: List[Dict[str, Any]] = []
        for p in programs:
            quote = _short_quote(p.get("text", ""), max_words=12)
            lines.append(f"- {p['program_id']} | {p['chunk_id']} | {quote}")
            citations_struct.append(
                {
                    "program_id": p["program_id"],
                    "program_name": p["program_name"],
                    "source": p["source"],
                    "chunk_id": p["chunk_id"],
                    "score": p["score"],
                    "quote": quote,
                }
            )

        return {
            "mode": mode,
            "answer": "\n".join(lines),
            "citations": citations_struct,
            "hits": hits_view,
        }

    # ----------------
    # ANSWER MODE (RAG)
    # ----------------
    context = _build_context(hits[:top_k])
    user_prompt = f"""QUESTION:
{query}

CONTEXT EXCERPTS:
{context}
"""
    answer = llm.chat(SYSTEM_PROMPT, user_prompt)

    citations: List[Dict[str, Any]] = []
    for score, payload in hits[:top_k]:
        citations.append(
            {
                "program_id": payload.get("program_id") or payload.get("doc_id") or "NA",
                "chunk_id": payload.get("chunk_id") or "NA",
                "quote": _short_quote(payload.get("text") or "", max_words=12),
                "score": float(score),
                "source": payload.get("source", "unknown"),
            }
        )

    return {
        "mode": mode,
        "answer": answer,
        "citations": citations,
        "hits": hits_view,
    }
