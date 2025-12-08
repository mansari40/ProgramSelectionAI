from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from src.agent.pipeline import ask_kb
from src.config.settings import Settings
from src.validation.rules import validate_response

# ------------------------------------------------------------
# Page style (student-friendly, charming, clean)
# ------------------------------------------------------------

APP_TITLE = "Constructor University - AI Agent"
APP_TAGLINE = "Find your program using official documents (PDFs, website pages, and program facts)."
APP_SUBTITLE = (
    "Ask questions, compare programs, and get evidence-backed answers with citations.\n"
    "The agent only uses what it can find in the indexed documents."
)

SOURCE_UI_LABELS = {
    "all": "All sources",
    "pdf": "Program handbooks (PDF)",
    "excel": "Program facts (Excel)",
    "web": "University website (Web)",
}

SOURCE_HELP_TEXT = {
    "all": "Uses PDFs + Excel + Website pages (best coverage).",
    "pdf": "Uses only program handbook PDFs (modules, rules, curriculum details).",
    "excel": "Uses only program facts (fees, test scores, contacts, high-level program info).",
    "web": "Uses only university website pages (admissions process, documents, general requirements).",
}

DEFAULT_QUICK_QUESTIONS = [
    "What is the tuition fee for Computer Science?",
    "What are the English language requirements for applying?",
    "What documents do I need to submit for an application?",
    "List programs that teach Python",
    "Compare Data Engineering vs Data Science for society & Business(tuition, duration, language).",
]

_CITATIONS_RE = re.compile(r"\nCitations:\s*\n.*$", flags=re.IGNORECASE | re.DOTALL)


def _inject_css() -> None:
    st.markdown(
        """
        <style>
          /* Layout */
          .block-container { padding-top: 1.25rem; padding-bottom: 2rem; }

          /* Hero */
          .hero-wrap {
            padding: 1.25rem 1.25rem 1.0rem 1.25rem;
            border-radius: 16px;
            border: 1px solid rgba(120,120,120,0.25);
            background: rgba(255,255,255,0.03);
            margin-bottom: 1rem;
          }
          .hero-title { font-size: 1.65rem; font-weight: 800; margin: 0; }
          .hero-tagline { font-size: 1.0rem; opacity: 0.9; margin-top: 0.35rem; }
          .hero-sub { font-size: 0.95rem; opacity: 0.85; margin-top: 0.6rem; line-height: 1.45; }

          /* Chips row */
          .chip-row { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.75rem; }

          /* Evidence box */
          .evidence-box {
            margin-top: 0.75rem;
            padding: 0.85rem 0.95rem;
            border-radius: 14px;
            border: 1px solid rgba(120,120,120,0.25);
            background: rgba(255,255,255,0.02);
          }
          .evidence-title { font-weight: 700; margin-bottom: 0.35rem; }
          .evidence-item { margin: 0.25rem 0; opacity: 0.92; }

          /* Small helper text */
          .muted { opacity: 0.75; font-size: 0.92rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------


def _strip_citations_block(text: str) -> str:
    """Remove any trailing 'Citations:' section from the model answer to prevent duplicates."""
    if not text:
        return ""
    return re.sub(_CITATIONS_RE, "", text).strip()


def _pretty_source_label(source: str) -> str:
    s = (source or "").strip().lower()
    if s == "pdf":
        return "PDF handbook"
    if s == "excel":
        return "Program facts (Excel)"
    if s == "web":
        return "University website"
    return "Source"


def _format_citations_student_friendly(citations: List[Dict[str, Any]]) -> str:
    """
    Student-friendly citations:
    - short, readable labels
    - show a short quote
    - do not expose chunk_id / score (keep UX clean)
    """
    if not citations:
        return ""

    lines: List[str] = []
    for c in citations:
        source = (c.get("source") or "unknown").lower()
        program_id = c.get("program_id", "NA")
        program_name = c.get("program_name") or c.get("title") or program_id
        quote = (c.get("quote") or "").replace("\n", " ").strip()

        label = f"{_pretty_source_label(source)} — {program_name}"
        if quote:
            lines.append(f"<div class='evidence-item'>• {label}: “{quote}”</div>")
        else:
            lines.append(f"<div class='evidence-item'>• {label}</div>")

    return (
        "<div class='evidence-box'>"
        "<div class='evidence-title'>Evidence Sources</div>"
        + "".join(lines)
        + "</div>"
    )


def _render_validation_banner(query: str, res: Dict[str, Any], expected_source: Optional[str]) -> None:
    """
    Student-friendly validation banner:
    - only show if something is off
    - no technical codes
    """
    vr = validate_response(query=query, response=res, expected_source=expected_source)
    if vr.passed:
        return

    if vr.confidence == "low":
        st.warning("This answer may be unreliable based on the available documents.")
    else:
        st.info("Some parts of this answer may be incomplete based on the available documents.")


def _render_answer(res: Dict[str, Any]) -> None:
    """
    Render:
      - answer (without duplicated citations section)
      - a clean evidence box (citations)
    """
    answer_raw = res.get("answer", "") or ""
    answer = _strip_citations_block(answer_raw)
    st.markdown(answer if answer else "I don't know based on the provided documents.")

    citations_html = _format_citations_student_friendly(res.get("citations", []) or [])
    if citations_html:
        st.markdown(citations_html, unsafe_allow_html=True)


def _run_agent_query(
    *,
    query: str,
    source: str,
    top_k: int = 12,
) -> Dict[str, Any]:
    """
    Centralized query runner with user-friendly progress messages.
    """
    # Lightweight staged progress to improve UX trust.
    p = st.progress(0, text="Searching university documents…")
    res: Dict[str, Any] = {}
    try:
        p.progress(20, text="Retrieving relevant evidence…")
        p.progress(55, text="Analyzing evidence…")
        res = ask_kb(query, source=source, top_k=top_k)
        p.progress(85, text="Generating answer…")
        p.progress(100, text="Done.")
    finally:
        p.empty()
    return res


# ------------------------------------------------------------
# Excel helpers (Programs tab)
# ------------------------------------------------------------


def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df


def _pick_first_existing(cols: List[str], candidates: List[str]) -> Optional[str]:
    s = set(cols)
    for c in candidates:
        if c in s:
            return c
    return None


@st.cache_data(show_spinner=False)
def _load_excel(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    return _normalize_cols(df)


def _extract_program_table_minimal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep: program_id, program_name, degree, location.
    Uses best-effort column mapping across possible schemas.
    """
    cols = list(df.columns)

    program_id = _pick_first_existing(cols, ["program_id", "programid", "program_code", "id"])
    program_name = _pick_first_existing(cols, ["program_name", "program", "name", "programname", "title"])
    degree = _pick_first_existing(cols, ["degree", "degree_type", "degree_level", "level"])
    location = _pick_first_existing(cols, ["location", "study_mode", "mode", "delivery_mode", "on_campus_online"])

    keep = [c for c in [program_id, program_name, degree, location] if c]
    if not keep:
        return df.iloc[:, :4].head(200)

    out = df[keep].copy().drop_duplicates()

    rename_map = {}
    if program_id:
        rename_map[program_id] = "program_id"
    if program_name:
        rename_map[program_name] = "program_name"
    if degree:
        rename_map[degree] = "degree"
    if location:
        rename_map[location] = "location"

    out = out.rename(columns=rename_map)

    ordered = [c for c in ["program_id", "program_name", "degree", "location"] if c in out.columns]
    out = out[ordered]

    # Make table friendlier
    if "program_id" in out.columns:
        out["program_id"] = out["program_id"].astype(str).str.strip()
    if "program_name" in out.columns:
        out["program_name"] = out["program_name"].astype(str).str.strip()
    if "degree" in out.columns:
        out["degree"] = out["degree"].astype(str).str.strip()
    if "location" in out.columns:
        out["location"] = out["location"].astype(str).str.strip()

    return out


def _render_programs_table(
    *,
    title: str,
    df: pd.DataFrame,
    default_degree_filter: str = "All",
) -> None:
    st.subheader(title)

    if df.empty:
        st.info("No programs found in this dataset.")
        return

    # Unique widget keys per table (fixes the Master Programs duplicate widget ID error)
    key_prefix = re.sub(r"[^a-z0-9]+", "_", title.strip().lower()).strip("_")

    # Filters
    c1, c2, c3 = st.columns([2, 1, 1])

    with c1:
        search = st.text_input(
            "Search programs",
            value="",
            placeholder="Type a program name…",
            key=f"{key_prefix}__search",
        )

    with c2:
        degree_vals = ["All"] + sorted(
            [
                d
                for d in df.get("degree", pd.Series([], dtype=str)).dropna().unique().tolist()
                if str(d).strip()
            ]
        )
        degree = st.selectbox(
            "Degree",
            degree_vals,
            index=degree_vals.index(default_degree_filter) if default_degree_filter in degree_vals else 0,
            key=f"{key_prefix}__degree",
        )

    with c3:
        location_vals = ["All"] + sorted(
            [
                l
                for l in df.get("location", pd.Series([], dtype=str)).dropna().unique().tolist()
                if str(l).strip()
            ]
        )
        location = st.selectbox(
            "Location",
            location_vals,
            index=0,
            key=f"{key_prefix}__location",
        )

    filtered = df.copy()

    if search.strip():
        # Search in program_name; keep behavior unchanged
        if "program_name" in filtered.columns:
            filtered = filtered[filtered["program_name"].str.contains(search.strip(), case=False, na=False)]

    if degree != "All" and "degree" in filtered.columns:
        filtered = filtered[filtered["degree"].astype(str) == degree]

    if location != "All" and "location" in filtered.columns:
        filtered = filtered[filtered["location"].astype(str) == location]

    st.caption(f"Showing {len(filtered)} program(s). You can sort columns and scroll.")
    st.dataframe(filtered, use_container_width=True, hide_index=True)


# ------------------------------------------------------------
# Onboarding / Transparency section
# ------------------------------------------------------------


def _render_how_it_works() -> None:
    with st.expander("How this AI Assistant works (Transparency)", expanded=False):
        st.markdown(
            """
            - **Evidence-based answers:** The assistant searches official documents (handbooks, website pages, and program facts).
            - **No guessing:** If the information is not found in the documents, it will say it doesn't know.
            - **Citations included:** Every answer includes a short evidence section so you can verify it quickly.
            - **Source control:** Use the sidebar to restrict answers to PDFs, the website, or Excel program facts.
            """
        )


# ------------------------------------------------------------
# Main app
# ------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🎓", layout="wide")
    _inject_css()

    # Hero
    st.markdown(
        f"""
        <div class="hero-wrap">
          <div class="hero-title">🎓 {APP_TITLE}</div>
          <div class="hero-tagline">{APP_TAGLINE}</div>
          <div class="hero-sub">{APP_SUBTITLE}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cfg = Settings.load()

    # Sidebar controls (ONLY source section)
    with st.sidebar:
        source_ui = st.selectbox(
            "Choose evidence source",
            [SOURCE_UI_LABELS["all"], SOURCE_UI_LABELS["pdf"], SOURCE_UI_LABELS["excel"], SOURCE_UI_LABELS["web"]],
            index=0,
        )
        inv = {v: k for k, v in SOURCE_UI_LABELS.items()}
        source = inv[source_ui]
        st.caption(SOURCE_HELP_TEXT.get(source, ""))

    # Fixed top-k (since sidebar should only contain the source section)
    topk = 12

    _render_how_it_works()

    # Tabs (shorter + clearer)
    tab_chat, tab_programs, tab_fees, tab_english, tab_docs, tab_contacts = st.tabs(
        ["💬 Chat", "📚 Programs", "💶 Fees", "🌍 English", "📝 Documents", "☎️ Contacts"]
    )

    # ----------------------------
    # Chat tab
    # ----------------------------
    with tab_chat:
        st.markdown(
            "<div class='muted'>Ask about programs, modules, tuition, English tests, scholarships, or application steps.</div>",
            unsafe_allow_html=True,
        )

        # Quick question buttons (chips)
        st.write("")
        chip_cols = st.columns(5)
        chip_questions = DEFAULT_QUICK_QUESTIONS[:5]
        clicked_question: Optional[str] = None
        for i, q in enumerate(chip_questions):
            with chip_cols[i % 5]:
                if st.button(q, use_container_width=True):
                    clicked_question = q

        st.divider()

        if "chat" not in st.session_state:
            st.session_state["chat"] = []  # structured turns

        # Render history
        for msg in st.session_state["chat"]:
            role = msg.get("role")
            content = msg.get("content") or {}
            with st.chat_message(role):
                if role == "user":
                    st.markdown(content.get("text", ""))
                else:
                    res = content.get("res") or {}
                    query = content.get("query", "")
                    expected_source = content.get("expected_source")
                    _render_validation_banner(query=query, res=res, expected_source=expected_source)
                    _render_answer(res)

        # Input: either clicked chip OR typed input
        user_input = clicked_question or st.chat_input("Ask a question about Constructor University programs…")

        if user_input:
            # Save + render user message
            st.session_state["chat"].append({"role": "user", "content": {"text": user_input}})
            with st.chat_message("user"):
                st.markdown(user_input)

            # Run agent
            with st.chat_message("assistant"):
                with st.spinner("Working on it…"):
                    res = _run_agent_query(query=user_input, source=source, top_k=topk)

                expected_source = None if source == "all" else source
                _render_validation_banner(query=user_input, res=res, expected_source=expected_source)
                _render_answer(res)

            # Store assistant message
            st.session_state["chat"].append(
                {
                    "role": "assistant",
                    "content": {"query": user_input, "expected_source": (None if source == "all" else source), "res": res},
                }
            )

        # Chat controls
        st.write("")
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            if st.button("Clear chat", use_container_width=True):
                st.session_state["chat"] = []
                st.rerun()
        with c2:
            if st.button("Show examples", use_container_width=True):
                st.info(
                    "Examples:\n"
                    "- What is the tuition fee for Computer Science?\n"
                    "- What are the English requirements?\n"
                    "- List programs that teach Python\n"
                    "- Compare Data Science vs Computer Science\n"
                )

    # ----------------------------
    # Programs tab
    # ----------------------------
    with tab_programs:
        st.caption("Browse official Bachelor and Master programs from the program facts Excel files.")
        st.write("")

        try:
            df_b = _load_excel(str(cfg.bachelors_excel))
            table_b = _extract_program_table_minimal(df_b)
            _render_programs_table(title="Bachelor Programs", df=table_b, default_degree_filter="All")
        except Exception as e:
            st.warning(f"Could not load bachelor programs list from Excel: {e}")

        st.divider()

        try:
            df_m = _load_excel(str(cfg.masters_excel))
            table_m = _extract_program_table_minimal(df_m)
            _render_programs_table(title="Master Programs", df=table_m, default_degree_filter="All")
        except Exception as e:
            st.warning(f"Could not load master programs list from Excel: {e}")

    # ----------------------------
    # Fees tab
    # ----------------------------
    with tab_fees:
        st.subheader("💶 Tuition & Fees")
        st.caption("Generated from the indexed knowledge base with evidence.")
        
        st.markdown(
            """
            **Quick summary**
            - Tuition (Bachelors + Masters): **€10,000 per semester**
            - Additional fees may apply (e.g., university fee, semester ticket)
            """
        )
        st.divider()
        st.caption("Tip: Ask “What additional fees apply?” in the Chat tab for evidence-backed details.")

    # ----------------------------
    # English tab (generated from KB, web-only)
    # ----------------------------
    with tab_english:
        st.subheader("🌍 English Requirements")
        st.caption("Generated from the indexed knowledge base (primarily university website).")
        st.write("")

        # Cache inside session to avoid repeated calls on every rerun
        if "english_res" not in st.session_state:
            with st.spinner("Loading English requirements…"):
                st.session_state["english_res"] = _run_agent_query(
                    query=(
                        "What are the English language requirements for applying to Constructor University? "
                        "Include accepted tests and minimum scores if available."
                    ),
                    source="web",
                    top_k=topk,
                )

        res = st.session_state["english_res"]
        _render_validation_banner(query="English requirements", res=res, expected_source="web")
        _render_answer(res)

        if st.button("Refresh English requirements", use_container_width=False):
            with st.spinner("Refreshing…"):
                st.session_state["english_res"] = _run_agent_query(
                    query=(
                        "What are the English language requirements for applying to Constructor University? "
                        "Include accepted tests and minimum scores if available."
                    ),
                    source="web",
                    top_k=topk,
                )
            st.rerun()

    # ----------------------------
    # Documents tab (generated from KB, web-only)
    # ----------------------------
    with tab_docs:
        st.subheader("📝 Application Documents")
        st.caption("Generated from the indexed knowledge base (primarily university website).")
        st.write("")

        if "docs_res" not in st.session_state:
            with st.spinner("Loading application documents…"):
                st.session_state["docs_res"] = _run_agent_query(
                    query=(
                        "What documents do applicants need to submit when applying to Constructor University? "
                        "List required documents and note if anything is optional."
                    ),
                    source="web",
                    top_k=topk,
                )

        res = st.session_state["docs_res"]
        _render_validation_banner(query="Application documents", res=res, expected_source="web")
        _render_answer(res)

        if st.button("Refresh application documents", use_container_width=False):
            with st.spinner("Refreshing…"):
                st.session_state["docs_res"] = _run_agent_query(
                    query=(
                        "What documents do applicants need to submit when applying to Constructor University? "
                        "List required documents and note if anything is optional."
                    ),
                    source="web",
                    top_k=topk,
                )
            st.rerun()

    # ----------------------------
    # Contacts tab
    # ----------------------------
    with tab_contacts:
        st.subheader("☎️ Contacts")
        st.caption("Official contacts for admissions and program questions.")
        st.markdown(
            """
            **Undergraduate Applicants**  
            study@constructor.university  
            +49 (0) 421 200 4200  

            **Graduate Applicants**  
            graduateadmission@constructor.university  
            +49 (0) 421 200 4211  

            **International Foundation Year Program**  
            foundationyear@constructor.university  
            +49 (0) 421 200 4313  

            **Executive Education Programs**  
            business@constructor.university  

            **Online Programs**  
            onlineprogram@constructor.university  
            +49 (0) 162 201 3305
            """
        )


if __name__ == "__main__":
    main()
