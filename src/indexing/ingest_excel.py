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
        return s == "" or s.lower() in {"na", "n/a", "-", "none"}
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


def _kv(label: str, value: str) -> str:
    if not value:
        return ""
    return f"- {label}: {value}"


def _build_sections(row: Dict[str, Any]) -> List[Tuple[str, str]]:
    """
    Returns a list of (section_name, text) for a program card.
    Keep sections compact and highly scannable for retrieval.
    """
    program_id = _field(row, "Program_ID")
    school = _field(row, "School_name")
    name = _field(row, "Program_name")
    abbr = _field(row, "Program_abbreviation")
    deg_level = _field(row, "Degree")  # e.g. Bachelor
    deg_type = _field(row, "Degree.1") if "Degree.1" in row else _field(row, "Degree")  # fallback
    location = _field(row, "Location")
    years = _field(row, "Years of study")
    credits = _field(row, "Program_credit")
    language = _field(row, "Lanugauge") or _field(row, "Language")
    link = _field(row, "Program_link")

    # Admissions/tests
    toefl_pbt = _field(row, "TOEFL_PBT")
    toefl_ibt = _field(row, "TOEFL_IBT")
    ielts = _field(row, "IELTS")
    sat = _field(row, "SAT")
    duolingo = _field(row, "Duolingo")
    sat_act = _field(row, "SAT/ACT")

    app_periods = _field(row, "Application_Periods")
    rolling_visa = _field(row, "Rolling_Admissions (Visa)")
    rolling_novisa = _field(row, "Rolling_Admissions (No Visa)")
    start_dates = _field(row, "Program_start_dates")

    # Fees / accommodation
    fees_sem = _field(row, "Fees_per_semester_euro")
    accom_fee = _field(row, "On-campus_accommodation_fee_euro")
    univ_fee = _field(row, "University_fee_euro")
    room_type = _field(row, "Room_type")
    accom_duration = _field(row, "Accomodation_duration")

    # Scholarship / contacts
    schol_avail = _field(row, "Scholorship_availability")
    schol_amount = _field(row, "Scholorship_amount_euro")
    sfs_email = _field(row, "Student Financial Services_Email")
    sfs_phone = _field(row, "Student Financial Services_Phone_number")
    student_services_email = _field(row, "Student_services_Email")

    # ---- Section 1: Overview ----
    overview_lines = [
        _kv("Program ID", program_id),
        _kv("Program name", name),
        _kv("Abbreviation", abbr),
        _kv("School", school),
        _kv("Degree level", deg_level),
        _kv("Degree type", deg_type),
        _kv("Location", location),
        _kv("Language", language),
        _kv("Years of study", years),
        _kv("Credits (ECTS)", credits),
        _kv("Program link", link),
    ]
    overview = "\n".join([x for x in overview_lines if x])

    # ---- Section 2: Admissions & start dates ----
    admissions_lines = [
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
        _kv("SAT/ACT note", sat_act),
    ]
    admissions = "\n".join([x for x in admissions_lines if x != ""])

    # ---- Section 3: Fees & accommodation ----
    fees_lines = [
        _kv("Fees per semester (EUR)", fees_sem),
        _kv("University fee (EUR)", univ_fee),
        _kv("On-campus accommodation fee (EUR)", accom_fee),
        _kv("Room type", room_type),
        _kv("Accommodation duration", accom_duration),
    ]
    fees = "\n".join([x for x in fees_lines if x])

    # ---- Section 4: Scholarships & contacts ----
    support_lines = [
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

    Key design:
    - program_id remains the raw Program_ID from the sheet
    - doc_id is scoped to avoid collisions: f"{excel_scope}::{program_id}"
    - chunk_id uses doc_id so rows from bachelor/master never overwrite each other in Qdrant
    """
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel not found: {excel_path}")

    df = pd.read_excel(excel_path)
    if df.empty:
        return []

    # Normalize column names (keep original too, but this avoids whitespace surprises)
    df.columns = [str(c).strip() for c in df.columns]

    chunks: List[Chunk] = []

    for _, row in df.iterrows():
        row_dict = row.to_dict()

        program_id = _clean(row_dict.get("Program_ID"))
        if not program_id:
            # Skip rows that cannot be identified reliably
            continue

        scoped_doc_id = f"{excel_scope}::{program_id}"

        title = f"{program_id} ({excel_scope})"
        program_name = _clean(row_dict.get("Program_name"))
        if program_name:
            title = f"{program_name} | {program_id} ({excel_scope})"

        sections = _build_sections(row_dict)

        for i, (section_name, section_text) in enumerate(sections):
            meta: Dict[str, Any] = {
                "title": title,
                "program_id": program_id,
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
