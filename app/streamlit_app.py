from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from src.agent.pipeline import ask_kb
from src.config.settings import Settings



# Helpers 

_CITATIONS_RE = re.compile(r"\nCitations:\s*\n.*$", flags=re.IGNORECASE | re.DOTALL)


def _strip_citations_block(text: str) -> str:
    """Remove any trailing 'Citations:' section from the model answer to prevent duplicates."""
    if not text:
        return ""
    return re.sub(_CITATIONS_RE, "", text).strip()


def _format_citations(citations: List[Dict[str, Any]]) -> str:
    """
    Render citations as a simple bullet list (student-friendly).
    No chunk_id / score display.
    """
    if not citations:
        return ""

    lines: List[str] = []
    for c in citations:
        source = (c.get("source") or "unknown").lower()
        program_id = c.get("program_id", "NA")
        program_name = c.get("program_name") or c.get("title") or program_id
        quote = (c.get("quote") or "").replace("\n", " ").strip()

        if source == "pdf":
            src_label = "PDF handbook"
        elif source == "excel":
            src_label = "Program facts (Excel)"
        elif source == "web":
            src_label = "University website"
        else:
            src_label = "Source"

        label = f"{src_label}: {program_name}"

        if quote:
            lines.append(f"- {label} — “{quote}”")
        else:
            lines.append(f"- {label}")

    return "\n".join(lines)


def _render_answer_with_citations(res: Dict[str, Any]) -> None:
    """Render a single answer + a single citations block (no duplicates)."""
    answer_raw = res.get("answer", "") or ""
    answer = _strip_citations_block(answer_raw)

    st.markdown(answer if answer else "I don't know based on the provided documents.")

    citations_md = _format_citations(res.get("citations", []) or [])
    if citations_md:
        st.markdown("\n\n**Citations**\n" + citations_md)



# Excel

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
    Only keep: program_name, degree, location (no program_id).
    Uses best-effort column mapping across possible schemas.
    """
    cols = list(df.columns)

    program_name = _pick_first_existing(cols, ["program_name", "program", "name", "programname", "title"])
    degree = _pick_first_existing(cols, ["degree", "degree_type", "degree_level", "level"])
    location = _pick_first_existing(cols, ["location", "study_mode", "mode", "delivery_mode", "on_campus_online"])

    keep = [c for c in [program_name, degree, location] if c]
    if not keep:
        
        return df.iloc[:, :3].head(200)

    out = df[keep].copy().drop_duplicates()

    
    rename_map = {}
    if program_name:
        rename_map[program_name] = "program_name"
    if degree:
        rename_map[degree] = "degree"
    if location:
        rename_map[location] = "location"

    out = out.rename(columns=rename_map)

    
    ordered = [c for c in ["program_name", "degree", "location"] if c in out.columns]
    return out[ordered]



# App

def main() -> None:
    st.set_page_config(
        page_title="Constructor University-Programs Selection AI Assistant",
        page_icon="🎓",
        layout="wide",
    )

    st.title("Constructor University-Programs Selection AI Assistant App")
    st.write(
        "Welcome to our Constructor University Program Selection AI Assistant.\n Ask questions about any program of your interest and I will try to provide helpful answers.\n For validation purposes, answers include citations at the end."
    )

    cfg = Settings.load()

    with st.sidebar:
        st.header("Data source (for Assistant tab)")
        source_ui = st.selectbox(
            "Use information from",
            ["All sources", "Program handbooks (PDF)", "Program facts (Excel)", "University website (Web)"],
            index=0,
        )
        source_map = {
            "All sources": "all",
            "Program handbooks (PDF)": "pdf",
            "Program facts (Excel)": "excel",
            "University website (Web)": "web",
        }
        source = source_map[source_ui]

        st.divider()
        st.caption("Examples:")
        st.code("What is the tuition fee for Computer Science?")
        st.code("Which modules are mandatory in Robotics and Intelligent Systems?")
        st.code("List programs that teach Python")

    tab_assistant, tab_programs, tab_fees, tab_english, tab_docs, tab_contacts = st.tabs(
        [
            "Assistant Chat",
            "Programs",
            "Tuition & Fees",
            "English Requirements",
            "Application Documents",
            "Contacts",
        ]
    )

    
    # Assistant Chat
    
    with tab_assistant:
        if "chat" not in st.session_state:
            st.session_state["chat"] = []

        for msg in st.session_state["chat"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_input = st.chat_input("Ask a question about Constructor University programs…")
        if user_input:
            st.session_state["chat"].append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                with st.spinner("Searching and generating an answer..."):
                    res = ask_kb(user_input, source=source, top_k=12)

                answer_raw = res.get("answer", "") or ""
                answer = _strip_citations_block(answer_raw)
                citations_md = _format_citations(res.get("citations", []) or [])

                final_md = answer if answer else "I don't know based on the provided documents."
                if citations_md:
                    final_md += "\n\n**Citations**\n" + citations_md

                st.markdown(final_md)
                st.session_state["chat"].append({"role": "assistant", "content": final_md})

    
    # Programs 
    
    with tab_programs:
        st.subheader("Bachelor Programs")
        try:
            df_b = _load_excel(str(cfg.bachelors_excel))
            table_b = _extract_program_table_minimal(df_b)
            st.dataframe(table_b, use_container_width=True, hide_index=True)
        except Exception as e:
            st.warning(f"Could not load bachelor programs list from Excel: {e}")

        st.subheader("Master Programs")
        try:
            df_m = _load_excel(str(cfg.masters_excel))
            table_m = _extract_program_table_minimal(df_m)
            st.dataframe(table_m, use_container_width=True, hide_index=True)
        except Exception as e:
            st.warning(f"Could not load master programs list from Excel: {e}")

    
    # Tuition & Fees 
    
    with tab_fees:
        st.subheader("Tuition & Fees")
        st.write(
            "Tuition for all the Bachelors and Masters programs are €10,000 per semester. "
            "Also, there are additional fees, e.g. semester ticket fee and university fee."
        )

    
    # English Requirements 
    
    with tab_english:
        st.subheader("English Requirements")
        st.caption("This section is generated from the indexed knowledge base (primarily the university website).")
        with st.spinner("Loading English requirements..."):
            res = ask_kb(
                "What are the English language requirements for applying to Constructor University? "
                "Include accepted tests and minimum scores if available.",
                source="web",
                top_k=12,
            )
        _render_answer_with_citations(res)

    
    # Application Documents 
    
    with tab_docs:
        st.subheader("Application Documents")
        st.caption("This section is generated from the indexed knowledge base (primarily the university website).")
        with st.spinner("Loading application document requirements..."):
            res = ask_kb(
                "What documents do applicants need to submit when applying to Constructor University? "
                "List required documents and note if anything is optional.",
                source="web",
                top_k=12,
            )
        _render_answer_with_citations(res)

    
    # Contacts
    
    with tab_contacts:
        
        st.markdown(
            """For any questions regarding admissions and programs, please reach out to the appropriate contact below:

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
