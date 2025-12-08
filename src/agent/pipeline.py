from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from src.config.settings import Settings
from src.llm.openai_client import OpenAIClient
from src.retrieval.program_index import load_program_index
from src.retrieval.program_resolver import resolve_program_ids
from src.retrieval.qdrant_store import get_client, search

# Max number of evidence items returned to Streamlit (Evidence Sources box)
MAX_EVIDENCE = 5

SYSTEM_PROMPT = """You are a retrieval-augmented assistant for Constructor University.

You MUST follow these rules:
1) Use ONLY the provided context excerpts as evidence.
2) Do NOT assume anything not present in the excerpts.
3) If the excerpts do not contain the answer, say: "I don't know based on the provided documents."
4) When the question compares two programs, you must show evidence for BOTH; if evidence for one side is missing, say so.
5) Fees terminology must be precise:
   - "Tuition fees per semester" and "University fee" are different fields.
   - Never label "University fee" as "tuition" unless the excerpt explicitly does so.
   - If tuition is missing but university fee exists, you MUST say tuition is not found.
"""

LIST_INTENT_RE = re.compile(
    r"^(list|show me|show|which programs|programs that|find programs|find|give me all|all programs)\b",
    re.IGNORECASE,
)

IDK = "I don't know based on the provided documents."

FEE_INTENT_RE = re.compile(
    r"\b(tuition|fee|fees|cost|price|payment|per semester|university fee)\b",
    re.IGNORECASE,
)

SCHOLARSHIP_INTENT_RE = re.compile(
    r"\b(scholarship|scholarships|financial aid|funding|grant)\b",
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
      None -> means "all"
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
    """Group by program_id/doc_id and keep the best-scoring hit per program."""
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


def _resolve_program_focus(query: str, cfg: Settings) -> List[str]:
    """
    Return a list of program_ids that the query is likely about.
    Uses Excel-based program index for name/abbr matching.
    """
    try:
        program_index = load_program_index(cfg)
        candidates = resolve_program_ids(query, program_index, max_results=3, min_score=0.50)
        return [c.program_id for c in candidates]
    except Exception:
        return []


def _looks_program_specific(query: str) -> bool:
    q = (query or "").lower()
    if re.search(r"\b(ast|dssb|csse|ris|de|mmda|det|qls|scm|mba|cs)\b", q):
        return True
    if "for " in q and ("master" in q or "msc" in q or "bachelor" in q or "bsc" in q):
        return True
    return False


def _is_fee_query(query: str) -> bool:
    return bool(FEE_INTENT_RE.search(query or ""))


def _is_scholarship_query(query: str) -> bool:
    return bool(SCHOLARSHIP_INTENT_RE.search(query or ""))


def _boost_score_for_query(query: str, text: str) -> float:
    q = (query or "").lower()
    t = (text or "").lower()
    bonus = 0.0

    # modules intent
    if "elective" in q:
        if "elective module" in t or "elective modules" in t or "mandatory elective" in t:
            bonus += 0.08
    if "core" in q:
        if "core module" in t or "core modules" in t:
            bonus += 0.08
    if "module" in q:
        if "module:" in t or "modules" in t:
            bonus += 0.03

    # fees intent
    if _is_fee_query(query):
        if "tuition fees per semester" in t or "fees per semester" in t:
            bonus += 0.12
        if "university fee" in t:
            bonus += 0.06

    # scholarship intent
    if _is_scholarship_query(query):
        if "scholarship availability" in t:
            bonus += 0.12
        if "scholarship amount" in t:
            bonus += 0.08

    return bonus


def _payload_program_id(payload: dict) -> str:
    return payload.get("program_id") or payload.get("doc_id") or "NA"


def _payload_section(payload: dict) -> str:
    sec = payload.get("section")
    if sec:
        return str(sec)
    meta = payload.get("meta") or {}
    if isinstance(meta, dict) and meta.get("section"):
        return str(meta.get("section"))
    return ""


def _dedupe_hits(hits: List[Tuple[float, dict]]) -> List[Tuple[float, dict]]:
    seen = set()
    out: List[Tuple[float, dict]] = []
    for score, payload in hits:
        cid = payload.get("chunk_id") or ""
        key = cid or (_payload_program_id(payload) + "::" + _snippet(payload.get("text") or "", 80))
        if key in seen:
            continue
        seen.add(key)
        out.append((score, payload))
    return out


def _limit_evidence(citations: List[Dict[str, Any]], max_items: int = MAX_EVIDENCE) -> List[Dict[str, Any]]:
    if not citations:
        return []
    return citations[:max_items]


def ask_kb(query: str, source: str = "all", top_k: int = 12) -> Dict[str, Any]:
    cfg = Settings.load()
    client = get_client(cfg)
    llm = OpenAIClient(cfg)

    mode = _mode_for_query(query)
    q_emb = llm.embed_texts([query])[0]
    src = _normalize_source(source)

    # Program focus resolution (AST/DSSB/etc.)
    program_ids = _resolve_program_focus(query, cfg)
    program_specific = _looks_program_specific(query) or bool(program_ids)

    # Retrieval sizing:
    if mode == "list":
        retrieve_k = min(max(top_k * 10, 120), 300)
        candidate_k = 1200
    else:
        retrieve_k = min(max(top_k * 15, 180), 300)
        candidate_k = 1500

    hits = search(
        client,
        cfg.qdrant_collection,
        q_emb,
        top_k=retrieve_k,
        source=src,
        candidate_k=candidate_k,
        query_filter=None,
    )

    # For UI/debug
    hits_view: List[Dict[str, Any]] = []
    for score, payload in hits:
        hits_view.append(
            {
                "score": float(score),
                "source": payload.get("source", "unknown"),
                "program_id": _payload_program_id(payload),
                "program_name": payload.get("program_name") or payload.get("title") or "NA",
                "chunk_id": payload.get("chunk_id") or "NA",
                "section": _payload_section(payload),
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
            return {"mode": mode, "answer": IDK, "citations": [], "hits": hits_view}

        keep_n = max(10, min(40, top_k * 4))
        programs = programs[:keep_n]

        lines = ["Programs found (best evidence per program):", ""]
        for i, p in enumerate(programs, start=1):
            lines.append(f"{i}. {p['program_name']} ({p['program_id']}) [{p['source']}]")

        # Build citations for Streamlit Evidence Sources
        citations_struct: List[Dict[str, Any]] = []
        for p in programs:
            citations_struct.append(
                {
                    "program_id": p["program_id"],
                    "program_name": p["program_name"],
                    "source": p["source"],
                    "chunk_id": p["chunk_id"],
                    "score": p["score"],
                    "quote": _short_quote(p.get("text", ""), max_words=12),
                    "section": _payload_section({"section": p.get("section", "")}),
                }
            )

        citations_struct = _limit_evidence(citations_struct)

        return {"mode": mode, "answer": "\n".join(lines), "citations": citations_struct, "hits": hits_view}

    # ----------------
    # ANSWER MODE (RAG)
    # ----------------
    chosen_hits: List[Tuple[float, dict]] = hits
    pid_set = set(program_ids) if program_ids else set()

    if pid_set:
        scoped = [(s, p) for (s, p) in hits if _payload_program_id(p) in pid_set]

        if not scoped and program_specific:
            return {"mode": mode, "answer": IDK, "citations": [], "hits": hits_view}

        chosen_hits = scoped if scoped else hits

    # Fee/scholarship safeguard:
    if pid_set and (_is_fee_query(query) or _is_scholarship_query(query)):
        focused_query = f"{query} tuition fees per semester university fee scholarship availability scholarship amount"
        fq_emb = llm.embed_texts([focused_query])[0]

        excel_hits = search(
            client,
            cfg.qdrant_collection,
            fq_emb,
            top_k=200,
            source="excel",
            candidate_k=800,
            query_filter=None,
        )

        excel_scoped = [(s, p) for (s, p) in excel_hits if _payload_program_id(p) in pid_set]

        preferred_sections = {"fees", "support", "overview"}
        excel_pref = [(s, p) for (s, p) in excel_scoped if _payload_section(p) in preferred_sections]

        rescored_excel: List[Tuple[float, dict]] = []
        for sc, payload in excel_pref:
            rescored_excel.append((float(sc) + _boost_score_for_query(query, payload.get("text") or ""), payload))
        rescored_excel.sort(key=lambda x: x[0], reverse=True)

        rescored_scoped: List[Tuple[float, dict]] = []
        for sc, payload in chosen_hits:
            rescored_scoped.append((float(sc) + _boost_score_for_query(query, payload.get("text") or ""), payload))
        rescored_scoped.sort(key=lambda x: x[0], reverse=True)

        chosen_hits = _dedupe_hits(rescored_excel + rescored_scoped)
    else:
        rescored: List[Tuple[float, dict]] = []
        for sc, payload in chosen_hits:
            rescored.append((float(sc) + _boost_score_for_query(query, payload.get("text") or ""), payload))
        rescored.sort(key=lambda x: x[0], reverse=True)
        chosen_hits = rescored

    context = _build_context(chosen_hits[:top_k])

    user_prompt = f"""QUESTION: {query}

CONTEXT EXCERPTS:
{context}
"""
    answer = llm.chat(SYSTEM_PROMPT, user_prompt)

    # Build citations for Streamlit Evidence Sources (then cap at MAX_EVIDENCE)
    citations: List[Dict[str, Any]] = []
    for score, payload in chosen_hits[:top_k]:
        citations.append(
            {
                "program_id": _payload_program_id(payload),
                "chunk_id": payload.get("chunk_id") or "NA",
                "quote": _short_quote(payload.get("text") or "", max_words=12),
                "score": float(score),
                "source": payload.get("source", "unknown"),
                "program_name": payload.get("program_name")
                or payload.get("title")
                or payload.get("program_id")
                or payload.get("doc_id")
                or "NA",
                "section": _payload_section(payload),
            }
        )

    citations = _limit_evidence(citations)

    return {"mode": mode, "answer": answer, "citations": citations, "hits": hits_view}
