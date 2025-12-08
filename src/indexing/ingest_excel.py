from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from src.schemas import Chunk
from src.utils.ids import make_chunk_id


def _is_empty(v: Any) -> bool:
    if v is None:
        return True

    if isinstance(v, float) and pd.isna(v):
        return True

    if isinstance(v, str):
        s = v.strip()
        if s == "":
            return True

        low = s.lower()
        if low in {"na", "n/a", "none"}:
            return True

        # common "missing" markers seen in your sheet
        # "-", "–", "€ -", "€ –", etc.
        s_no_eur = s.replace("€", "").strip()
        if s_no_eur in {"-", "–"}:
            return True

    return False


def _clean(v: Any) -> str:
    if _is_empty(v):
        return ""
    s = str(v).strip()
    # normalize internal whitespace
    s = " ".join(s.split())
    return s


def _field(row: Dict[str, Any], key: str) -> str:
    return _clean(row.get(key))


def _first(row: Dict[str, Any], keys: List[str]) -> str:
    for k in keys:
        val = _field(row, k)
        if val:
            return val
    return ""


def _kv(label: str, value: str) -> str:
    if not value:
        return ""
    return f"- {label}: {value}"


def _build_sections(row: Dict[str, Any], excel_scope: str) -> List[Tuple[str, str]]:
    """
    Returns a list of (section_name, text) for a program card.
    Design goals:
    - keep sections short and scannable
    - ensure tuition/scholarships are in the OVERVIEW (and also fees/support)
    - be robust to column naming variations
    """
    # Core identity
    program_id = _first(row, ["Program_ID", "Program ID"])
    school = _first(row, ["School_name", "School name", "School"])
    name = _first(row, ["Program_name", "Program name", "Program"])
    abbr = _first(row, ["Program_abbreviation", "Program abbreviation", "Abbreviation"])

    # Degree/structure
    deg_level = _first(row, ["Degree", "Degree_level", "Degree level"])  # e.g., Master
    deg_type = _first(row, ["Degree.1", "Degree_type", "Degree type"])   # e.g., MSc
    location = _first(row, ["Location"])
    years = _first(row, ["Years of study", "Years_of_study", "Duration", "Years"])
    credits = _first(row, ["Program_credit", "Program credit", "Credits", "ECTS"])
    language = _first(row, ["Lanugauge", "Language"])  # keep your sheet typo as fallback
    link = _first(row, ["Program_link", "Program link", "Link", "URL"])

    # Admissions/tests
    toefl_pbt = _first(row, ["TOEFL_PBT", "TOEFL PBT"])
    toefl_ibt = _first(row, ["TOEFL_IBT", "TOEFL iBT", "TOEFL IBT"])
    ielts = _first(row, ["IELTS"])
    sat = _first(row, ["SAT"])
    duolingo = _first(row, ["Duolingo"])
    gre = _first(row, ["GRE"])
    sat_act = _first(row, ["SAT/ACT", "SAT_ACT", "SAT/ACT note"])

    app_periods = _first(row, ["Application_Periods", "Application periods"])
    rolling_visa = _first(row, ["Rolling_Admissions (Visa)", "Rolling admissions (Visa)"])
    rolling_novisa = _first(row, ["Rolling_Admissions (No Visa)", "Rolling admissions (No Visa)"])
    start_dates = _first(row, ["Program_start_dates", "Program start dates"])

    # Fees / accommodation (IMPORTANT FIX: tuition column name)
    tuition_sem = _first(
        row,
        [
            "Tuition_fees_per_semester_euro",
            "Tuition fees per semester (EUR)",
            "Tuition_fees_per_semester_(EUR)",
            "Fees_per_semester_euro",  # backward-compat fallback
            "Fees per semester (EUR)",
        ],
    )
    accom_fee = _first(row, ["On-campus_accommodation_fee_euro", "On-campus accommodation fee (EUR)"])
    univ_fee = _first(row, ["University_fee_euro", "University fee (EUR)"])
    room_type = _first(row, ["Room_type", "Room type"])
    accom_duration = _first(row, ["Accomodation_duration", "Accommodation duration", "Accomodation duration"])

    # Scholarship / contacts
    schol_avail = _first(row, ["Scholorship_availability", "Scholarship availability"])
    schol_amount = _first(row, ["Scholorship_amount_euro", "Scholarship amount (EUR)"])
    sfs_email = _first(row, ["Student Financial Services_Email", "Student Financial Services email"])
    sfs_phone = _first(row, ["Student Financial Services_Phone_number", "Student Financial Services phone"])
    student_services_email = _first(row, ["Student_services_Email", "Student services email", "Admissions email"])

    # ---- Section 1: Overview (include tuition + scholarship to make retrieval easy) ----
    overview_lines = [
        _kv("Program ID", program_id),
        _kv("Program name", name),
        _kv("Abbreviation", abbr),
        _kv("Scope", excel_scope),
        _kv("School", school),
        _kv("Degree level", deg_level),
        _kv("Degree type", deg_type),
        _kv("Location", location),
        _kv("Language", language),
        _kv("Years of study", years),
        _kv("Credits (ECTS)", credits),
        _kv("Tuition fees per semester (EUR)", tuition_sem),
        _kv("University fee (EUR)", univ_fee),
        _kv("Scholarship availability", schol_avail),
        _kv("Scholarship amount (EUR)", schol_amount),
        _kv("Program link", link),
    ]
    overview = "\n".join([x for x in overview_lines if x])

    # ---- Section 2: Admissions & start dates ----
    admissions_lines = [
        _kv("Program ID", program_id),
        _kv("Program name", name),
        _kv("Abbreviation", abbr),
        _kv("Scope", excel_scope),
        _kv("Application periods", app_periods),
        _kv("Rolling admissions (Visa)", rolling_visa),
        _kv("Rolling admissions (No Visa)", rolling_novisa),
        _kv("Program start dates", start_dates),
        "",
        "Tests / requirements:",
        _kv("TOEFL PBT", toefl_pbt),
        _kv("TOEFL iBT", toefl_ibt),
        _kv("IELTS", ielts),
        _kv("SAT", sat),
        _kv("Duolingo", duolingo),
        _kv("GRE", gre),
        _kv("SAT/ACT note", sat_act),
    ]
    admissions = "\n".join([x for x in admissions_lines if x != ""])

    # ---- Section 3: Fees & accommodation ----
    fees_lines = [
        _kv("Program ID", program_id),
        _kv("Program name", name),
        _kv("Abbreviation", abbr),
        _kv("Scope", excel_scope),
        _kv("Tuition fees per semester (EUR)", tuition_sem),
        _kv("University fee (EUR)", univ_fee),
        _kv("On-campus accommodation fee (EUR)", accom_fee),
        _kv("Room type", room_type),
        _kv("Accommodation duration", accom_duration),
    ]
    fees = "\n".join([x for x in fees_lines if x])

    # ---- Section 4: Scholarships & contacts ----
    support_lines = [
        _kv("Program ID", program_id),
        _kv("Program name", name),
        _kv("Abbreviation", abbr),
        _kv("Scope", excel_scope),
        _kv("Scholarship availability", schol_avail),
        _kv("Scholarship amount (EUR)", schol_amount),
        _kv("Student Financial Services email", sfs_email),
        _kv("Student Financial Services phone", sfs_phone),
        _kv("Student services email", student_services_email),
    ]
    support = "\n".join([x for x in support_lines if x])

    sections: List[Tuple[str, str]] = []
    if overview:
        sections.append(("overview", overview))
    if admissions:
        sections.append(("admissions", admissions))
    if fees:
        sections.append(("fees", fees))
    if support:
        sections.append(("support", support))

    return sections


def ingest_excel_as_program_chunks(
    excel_path: Path,
    excel_scope: str,  # "bachelor" or "master"
    source: str = "excel",
) -> List[Chunk]:
    """
    Reads an Excel file and converts each row into multiple small "program card" chunks.

    Design:
    - program_id remains raw Program_ID from the sheet
    - doc_id is scoped: f"{excel_scope}::{program_id}"
    - chunk_id uses scoped doc_id => bachelor/master rows never collide in Qdrant
    """
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel not found: {excel_path}")

    df = pd.read_excel(excel_path)
    if df.empty:
        return []

    # Normalize column names: strip whitespace to match your header reliably
    df.columns = [str(c).strip() for c in df.columns]

    chunks: List[Chunk] = []

    for _, row in df.iterrows():
        row_dict = row.to_dict()

        program_id = _clean(row_dict.get("Program_ID"))
        if not program_id:
            continue

        scoped_doc_id = f"{excel_scope}::{program_id}"

        program_name = _clean(row_dict.get("Program_name"))
        title = f"{program_name}" if program_name else program_id

        sections = _build_sections(row_dict, excel_scope=excel_scope)

        for i, (section_name, section_text) in enumerate(sections):
            meta: Dict[str, Any] = {
                "title": title,
                "program_id": program_id,
                "program_name": program_name or title,
                "excel_scope": excel_scope,
                "section": section_name,
                "source_file": excel_path.name,
            }

            chunks.append(
                Chunk(
                    source=source,
                    doc_id=scoped_doc_id,
                    chunk_id=make_chunk_id(source, scoped_doc_id, i),
                    text=section_text,
                    meta=meta,
                )
            )

    return chunks
